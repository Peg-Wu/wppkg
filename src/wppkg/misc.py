import os
import re
import json
import logging
import hashlib
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from typing import Union, List, Optional, Literal

logger = logging.getLogger(__name__)


# Define a context manager to suppress output
class suppress_output:
    def __enter__(self):
        self._stdout = os.dup(1)
        self._stderr = os.dup(2)
        self._null = os.open(os.devnull, os.O_RDWR)
        os.dup2(self._null, 1)
        os.dup2(self._null, 2)
        return self

    def __exit__(self, *args):
        # First restore the original file descriptors
        os.dup2(self._stdout, 1)
        os.dup2(self._stderr, 2)
        # Then close all our saved descriptors
        os.close(self._stdout)
        os.close(self._stderr)
        os.close(self._null)


def remove_non_alphanumeric(input_string):
    return re.sub(r'[^a-zA-Z0-9]', '', input_string)


def get_string_md5(
    s: str, 
    encoding: str = "utf-8"
):
    md5_obj = hashlib.md5()
    md5_obj.update(s.encode(encoding=encoding))
    return md5_obj.hexdigest()


def read_json(
    fpath: str, 
    convert_to_easydict: bool = True
) -> dict:
    with open(fpath, "r") as f:
        obj = json.load(f)
    if convert_to_easydict:
        try:
            from easydict import EasyDict as edict
            obj = edict(obj)
        except ImportError:
            logger.warning(
                "easydict is not installed, falling back to standard dictionary - install with: pip install easydict"
            )
    return obj


def write_json(obj: dict, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))


def debugpy_header(port: int = 5678):
    import debugpy
    try:
        # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
        debugpy.listen(("localhost", port))
        print("Waiting for debugger attach ...")
        debugpy.wait_for_client()
    except Exception as e:
        print("Debugger attach failed ...")


def generate_default_debugpy_config(
    port: int = 5678,
    save_dir: str = "./"
):
    config_file = Path(__file__).resolve().parent / "configs" / "debugpy_config" / "launch.json"
    cfg = read_json(config_file, convert_to_easydict=False)
    cfg["configurations"][-1]["connect"]["port"] = port
    save_path = Path(save_dir) / ".vscode" / "launch.json"
    write_json(cfg, save_path)


def generate_default_deepspeed_config(
    config_name: Literal["zero1", "zero2", "zero2_offload", "zero3", "zero3_offload"],
    save_path: str
) -> None:
    assert Path(save_path).suffix.lower() == ".json", "Invalid path: must end with .json"

    config_file = Path(__file__).resolve().parent / "deepspeed_config" / (config_name + ".json")

    write_json(read_json(config_file, convert_to_easydict=False), save_path)


def get_sorted_indices_in_array_1d(
    arr: np.ndarray,
    ignore_zero_values: bool = False,
    descending: bool = True,
    **argsort_kwargs
) -> np.ndarray:
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape: {arr.shape}")
    
    if not ignore_zero_values:
        sorted_indices = np.argsort(arr, **argsort_kwargs)
    else:
        non_zero_indices = np.nonzero(arr)[0]
        non_zero_values = arr[non_zero_indices]
        sorted_indices = non_zero_indices[np.argsort(non_zero_values, **argsort_kwargs)]
    return sorted_indices[::-1] if descending else sorted_indices


def get_sorted_indices_in_array_2d_by_row(
    arr: np.ndarray,
    ignore_zero_values: bool = False,
    descending: bool = True,
    stable: Optional[bool] = None,
    n_jobs: int = 1,
    enable_tqdm: bool = True,
    **argsort_kwargs
) -> Union[np.ndarray, List[np.ndarray]]:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape: {arr.shape}")

    if not ignore_zero_values:
        sorted_indices = np.argsort(arr, axis=1, **argsort_kwargs)
        return sorted_indices[:, ::-1] if descending else sorted_indices
    else:
        sorted_indices = Parallel(n_jobs=n_jobs)(
            delayed(get_sorted_indices_in_array_1d)(
                arr=row, 
                ignore_zero_values=ignore_zero_values, 
                descending=descending,
                stable=stable,
                **argsort_kwargs
            ) for row in tqdm(arr, disable=not enable_tqdm)
        )
        return sorted_indices