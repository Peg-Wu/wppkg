import torch
import multiprocessing as mp

from torch import nn
from pathlib import Path
from collections.abc import Mapping
from transformers import BatchEncoding
from dataclasses import dataclass, field
from .utils import read_json, write_json
from huggingface_hub import snapshot_download
from typing import Optional, Union, Literal, List, Any


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
):
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


def generate_default_deepspeed_config(
    config_name: Literal["zero1", "zero2", "zero2_offload", "zero3", "zero3_offload"],
    save_path: str
):
    assert Path(save_path).suffix.lower() == ".json", "Invalid path: must end with .json"

    config_file = Path(__file__).resolve().parent / "deepspeed_config" / (config_name + ".json")

    write_json(read_json(config_file, convert_to_easydict=False), save_path)


@dataclass
class PaddingMixin:
    # If attention_mask is not present in your dataset, 
    # we will initialize it as all ones, with the padded token positions set to 0.
    return_attention_mask: bool = True

    # If multiple input sequences need to be padded, their lengths must be the same.
    model_input_names_need_pad: list[str] = field(default_factory=lambda: ["input_ids"])
    pad_token_id: list[int] = field(default_factory=lambda: [0])
    padding_side: Literal["right", "left"] = "right"

    def __post_init__(self):
        if len(self.model_input_names_need_pad) != len(self.pad_token_id):
            raise ValueError("Each sequence needs to have its own pad_token_id specified.")

    def pad(self, batch_data: list[dict[str, Any]]) -> BatchEncoding:
        max_length = max(len(one_data[self.model_input_names_need_pad[0]]) for one_data in batch_data)

        batch_outputs = {}
        for one_data in batch_data:
            outputs = self._pad(one_data, max_length)

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)
        
        return BatchEncoding(batch_outputs, tensor_type="pt")
    
    def _pad(
        self, 
        one_data: dict[str, Any],
        max_length: Optional[int] = None
    ) -> dict[str, Any]:
        required_input = one_data[self.model_input_names_need_pad[0]]

        # Initialize attention mask if not present.
        if self.return_attention_mask and "attention_mask" not in one_data:
            one_data["attention_mask"] = [1] * len(required_input)

        difference = max_length - len(required_input)

        # Pad the attention_mask first.
        if self.return_attention_mask:
            if self.padding_side == "right":
                one_data["attention_mask"] = one_data["attention_mask"] + [0] * difference
            elif self.padding_side == "left":
                one_data["attention_mask"] = [0] * difference + one_data["attention_mask"]
            else:
                raise ValueError(f"Invalid padding strategy: {self.padding_side}")

        # Pad keys in model_input_names_need_pad
        for key, pad_token_id in zip(self.model_input_names_need_pad, self.pad_token_id):
            if self.padding_side == "right":
                one_data[key] = one_data[key] + [pad_token_id] * difference
            elif self.padding_side == "left":
                one_data[key] = [pad_token_id] * difference + one_data[key]
            else:
                raise ValueError(f"Invalid padding strategy: {self.padding_side}")

        return one_data
    
    def _get_pad_token_id(self, model_input_name: str) -> int:
        return self.pad_token_id[self.model_input_names_need_pad.index(model_input_name)]
    

# Modified from transformers.DataCollatorForLanguageModeling
@dataclass
class DataCollatorForLanguageModeling(PaddingMixin):
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        mask_replace_prob (`float`, *optional*, defaults to 0.8):
            The probability with which masked tokens are replaced by the tokenizer's mask token (e.g., `[MASK]`).
            Defaults to 0.8, meaning 80% of the masked tokens will be replaced with `[MASK]`.
            Only works when `mlm` is set to `True`.
        random_replace_prob (`float`, *optional*, defaults to 0.1):
            The probability with which masked tokens are replaced by random tokens from the tokenizer's vocabulary.
            Defaults to 0.1, meaning 10% of the masked tokens will be replaced with random tokens. The remaining
            masked tokens (1 - mask_replace_prob - random_replace_prob) are left unchanged.
            Only works when `mlm` is set to `True`.
        seed (`int`, *optional*):
            The seed to use for the random number generator for masking. If not provided, the global RNG will be used.
    
    **Example:**
        >>> from wppkg.dl import DataCollatorForLanguageModeling
        >>> batch_list = [
        ...     {"input_ids": [101, 2054, 2003, 102]},
        ...     {"input_ids": [101, 1045, 2000, 2070, 102]},
        ... ]
        >>> datacollator = DataCollatorForLanguageModeling(mlm=True)
        >>> batch = datacollator(batch_list)
    
    <Example Options and Expectations>

    1. Default Behavior:
        - `mask_replace_prob=0.8`, `random_replace_prob=0.1`.
        - Expect 80% of masked tokens replaced with `[MASK]`, 10% replaced with random tokens, and 10% left unchanged.

    2. All masked tokens replaced by `[MASK]`:
        - `mask_replace_prob=1.0`, `random_replace_prob=0.0`.
        - Expect all masked tokens to be replaced with `[MASK]`. No tokens are left unchanged or replaced with random tokens.

    3. No `[MASK]` replacement, only random tokens:
        - `mask_replace_prob=0.0`, `random_replace_prob=1.0`.
        - Expect all masked tokens to be replaced with random tokens. No `[MASK]` replacements or unchanged tokens.

    4. Balanced replacement:
        - `mask_replace_prob=0.5`, `random_replace_prob=0.4`.
        - Expect 50% of masked tokens replaced with `[MASK]`, 40% replaced with random tokens, and 10% left unchanged.

    Note:
        The sum of `mask_replace_prob` and `random_replace_prob` must not exceed 1. If their sum is less than 1, the
        remaining proportion will consist of masked tokens left unchanged.
    """

    mlm: bool = True
    mlm_probability: Optional[float] = 0.15
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
    seed: Optional[int] = None

    # MLM/CLM is supported for only one sequence.
    model_input_names_need_mlm_or_clm: str = "input_ids"
    mask_token_id: int = 103

    # Cache all special token indices. (mlm task needed!)
    all_special_token_ids: list[int] = field(default_factory=lambda: [0])

    # Cache vocab_size for `model_input_names_need_mlm_or_clm` sequence. (mlm task needed, random replace token!)
    vocab_size: int = 30000

    def __post_init__(self):
        super().__post_init__()
        if self.mlm:
            if self.mlm_probability is None or self.mlm_probability < 0 or self.mlm_probability > 1:
                raise ValueError("mlm_probability should be between 0 and 1.")
            self.mlm_probability = float(self.mlm_probability)
        if self.mask_replace_prob + self.random_replace_prob > 1:
            raise ValueError("The sum of mask_replace_prob and random_replace_prob should not exceed 1")
        if self.mask_replace_prob < 0 or self.mask_replace_prob > 1:
            raise ValueError("mask_replace_prob should be between 0 and 1.")
        if self.random_replace_prob < 0 or self.random_replace_prob > 1:
            raise ValueError("random_replace_prob should be between 0 and 1.")

        self.mask_replace_prob = float(self.mask_replace_prob)
        self.random_replace_prob = float(self.random_replace_prob)
        self.generator = None

    def get_generator(self, seed):
        return torch.Generator().manual_seed(seed)

    def create_rng(self):
        if mp.current_process().name == "MainProcess":
            # If we are in the main process, we create a generator object with the seed
            self.generator = self.get_generator(self.seed)
        else:
            # If we are in a worker process (i.e using multiprocessing), we need to set a unique seed for each
            # worker's generator, generated as the main seed + the worker's ID.
            # (https://pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading)
            # Only PyTorch DataLoader allows us to access the worker ID, and so we check for this.
            # For other frameworks, we will throw an error.
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                error_string = (
                    "Worker process information is not available for seeding the generator. This may be because",
                    "you are using multiprocessing without using a PyTorch DataLoader. The `seed` parameter can",
                    "only be used when using multiprocessing with a PyTorch DataLoader. Please either use a",
                    "single process or use a PyTorch DataLoader with multiple workers.",
                )
                raise ValueError(error_string)

            self.generator = self.get_generator(self.seed + worker_info.id)

    def get_special_tokens_mask(self, token_ids: list[int]) -> list[int]:
        all_special_token_ids = self.all_special_token_ids  # cache the property
        special_tokens_mask = [1 if token in all_special_token_ids else 0 for token in token_ids]
        return special_tokens_mask

    def torch_mask_tokens(self, inputs: Any) -> tuple[Any, Any]:
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # Special tokens will not be masked.
        special_tokens_mask = [
            self.get_special_tokens_mask(val) for val in labels.tolist()
        ]
        no_mask_mask = (
            special_tokens_mask.bool()
            if isinstance(special_tokens_mask, torch.Tensor)
            else torch.tensor(special_tokens_mask, dtype=torch.bool)
        )
        probability_matrix.masked_fill_(no_mask_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix, generator=self.generator).bool()
        labels[~masked_indices] = -100

        # mask_replace_prob% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, self.mask_replace_prob), generator=self.generator).bool()
            & masked_indices
        )
        inputs[indices_replaced] = self.mask_token_id

        if self.mask_replace_prob == 1 or self.random_replace_prob == 0:
            return inputs, labels

        remaining_prob = 1 - self.mask_replace_prob
        # scaling the random_replace_prob to the remaining probability for example if
        # mask_replace_prob = 0.8 and random_replace_prob = 0.1,
        # then random_replace_prob_scaled = 0.1 / 0.2 = 0.5
        random_replace_prob_scaled = self.random_replace_prob / remaining_prob

        # random_replace_prob% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, random_replace_prob_scaled), generator=self.generator).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long, generator=self.generator)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time ((1-random_replace_prob-mask_replace_prob)% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def torch_call(self, batch_data: list[dict[str, Any]]) -> dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.

        if self.seed and self.generator is None:
            # If we have a seed, we need to create a generator object. Subsequent calls to this function will use the same generator.
            # If no seed supplied, we will use the global RNG
            self.create_rng()
        
        assert isinstance(batch_data[0], Mapping), (
            "This data collator should be used with a dataset having items that are dictionaries."
        )

        batch = self.pad(batch_data)

        seq_for_mlm_or_clm = self.model_input_names_need_mlm_or_clm
        if self.mlm:
            batch[seq_for_mlm_or_clm], batch["labels"] = self.torch_mask_tokens(
                batch[seq_for_mlm_or_clm]
            )
        else:
            labels = batch[seq_for_mlm_or_clm].clone()
            pad_token_id = self._get_pad_token_id(seq_for_mlm_or_clm)
            labels[labels == pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def __call__(self, batch_data: list[dict[str, Any]]) -> dict[str, Any]:
        return self.torch_call(batch_data)


# Modified from transformers.DataCollatorWithPadding
@dataclass
class DataCollatorWithPadding(PaddingMixin):
    def __call__(self, batch_data: list[dict[str, Any]]) -> dict[str, Any]:
        assert isinstance(batch_data[0], Mapping), (
            "This data collator should be used with a dataset having items that are dictionaries."
        )

        batch = self.pad(batch_data)

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch
