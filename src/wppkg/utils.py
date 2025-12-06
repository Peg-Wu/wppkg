import os
import json
import logging
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from typing import Union, List, Optional

logger = logging.getLogger(__name__)


def setup_logging_basic(
    main_process_level: Union[str, int] = logging.INFO,
    other_process_level: Union[str, int] = logging.WARN,
    local_rank: int = -1
):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=main_process_level if local_rank in [-1, 0] else other_process_level,
    )


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


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]