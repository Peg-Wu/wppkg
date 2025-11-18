from torch import nn
from pathlib import Path
from .utils import read_json, write_json
from huggingface_hub import snapshot_download
from typing import Optional, Union, Literal, List


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
    config_name: Literal["zero2", "zero2_offload", "zero3", "zero3_offload"],
    save_path: str
):
    assert Path(save_path).suffix.lower() == ".json", "Invalid path: must end with .json"

    config_file = Path(__file__).resolve().parent / "ds_config" / (config_name + ".json")

    write_json(read_json(config_file, convert_to_easydict=False), save_path)