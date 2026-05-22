import torch
from torch import nn
from pathlib import Path
from typing import Optional, Union, List
from huggingface_hub import snapshot_download


def get_nb_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    r"""
    Returns the number of trainable parameters and the number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "element_size"):
                num_bytes = param.element_size()
            elif not hasattr(param, "quant_storage"):
                num_bytes = 1
            else:
                num_bytes = param.quant_storage.itemsize
            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


# Copied from peft.peft_model
def print_trainable_parameters(model: nn.Module) -> None:
    """
    Prints the number of trainable parameters in the model.

    Note: print_trainable_parameters() uses get_nb_trainable_parameters() which is different from
    num_parameters(only_trainable=True) from huggingface/transformers. get_nb_trainable_parameters() returns
    (trainable parameters, all parameters) of the Peft Model which includes modified backbone transformer model.
    For techniques like LoRA, the backbone transformer model is modified in place with LoRA modules. However, for
    prompt tuning, the backbone transformer model is unmodified. num_parameters(only_trainable=True) returns number
    of trainable parameters of the backbone transformer model which can be different.
    """
    trainable_params, all_param = get_nb_trainable_parameters(model)

    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
    )


def hf_download(
    repo_id: str, 
    repo_type: Optional[str] = None,  # model, dataset, space
    allow_patterns: Optional[Union[List[str], str]] = None,
    ignore_patterns: Optional[Union[List[str], str]] = None,
    local_dir: Union[str, Path, None] = None,
    token: Optional[Union[bool, str]] = None,
    max_workers: int = 8,
    endpoint: Optional[str] = "https://hf-mirror.com"  # or https://huggingface.co
) -> None:
    r"""Download huggingface repo files.

    Args:
        repo_id (`str`):
            A user or an organization name and a repo name separated by a `/`.
        repo_type (`str`, *optional*):
            Set to `"dataset"` or `"space"` if downloading from a dataset or space,
            `None` or `"model"` if downloading from a model. Default is `None`.
        allow_patterns (`List[str]` or `str`, *optional*):
            If provided, only files matching at least one pattern are downloaded.
        ignore_patterns (`List[str]` or `str`, *optional*):
            If provided, files matching any of the patterns are not downloaded.
        local_dir (`str` or `Path`, *optional*):
            If provided, the downloaded files will be placed under this directory.
        token (`str`, `bool`, *optional*):
            A token to be used for the download.
                - If `True`, the token is read from the HuggingFace config folder.
                - If a string, it's used as the authentication token.
        max_workers (`int`, *optional*):
            Number of concurrent threads to download files (1 thread = 1 file download).
            Defaults to 8.
        endpoint (`str`, *optional*):
            Endpoint of the Hub. Defaults to <https://hf-mirror.com>.
    """
    print(
        snapshot_download(
            repo_id=repo_id, 
            repo_type=repo_type,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            local_dir=local_dir,
            token=token,
            max_workers=max_workers,
            endpoint=endpoint,
            library_name="hf"
        )
    )


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, name: list[str]) -> None:
        self.name = name
        self.data = [0.0] * len(name)  # initialize to zero
        self.add_times = 0  # track number of times `add` is called

    def add(self, *args) -> None:
        self.data = [a + float(b) for a, b in zip(self.data, args)]
        self.add_times += 1
    
    def mean(self) -> None:
        self.data = [a / self.add_times for a in self.data]

    def reset(self) -> None:
        self.data = [0.0] * len(self.data)
        self.add_times = 0
    
    def to_dict(self) -> dict:
        return {name: data for name, data in zip(self.name, self.data)}

    def __getitem__(self, key: Union[int, str]) -> float:
        # key is int → index lookup
        if isinstance(key, int):
            return self.data[key]
        
        # key is str → name lookup
        if isinstance(key, str):
            try:
                idx = self.name.index(key)
            except ValueError:
                raise KeyError(f"'{key}' not found in accumulator names {self.name}")
            return self.data[idx]
        
        raise TypeError(f"Invalid key type {type(key)}, expected int or str.")
    

class NoRoPE(nn.Module):
    """
    A drop-in replacement for LlamaRotaryEmbedding that always returns:
      cos = all ones, sin = all zeros
    of shape (1, seq_len, head_dim), so rotary has no effect.
    """

    def __init__(self, head_dim: int):
        super().__init__()
        # head_dim = config.hidden_size // config.num_attention_heads
        self.head_dim = head_dim

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, position_ids: torch.LongTensor):
        r"""
            - hidden_states: (batch_size, seq_len, hidden_dim)
            - position_ids: (1, seq_len)
        """
        _batch_size, seq_len, _hidden_dim = hidden_states.shape

        # Create cos = ones, sin = zeros
        #   shape --> (1, seq_len, head_dim)
        cos = hidden_states.new_ones(1, seq_len, self.head_dim)
        sin = hidden_states.new_zeros(1, seq_len, self.head_dim)
        return cos, sin