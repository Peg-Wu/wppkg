import os
import json
import logging
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from typing import Union, List, Optional

logger = logging.getLogger(__name__)


def _configure_logger(
    logger: logging.Logger,
    log_file: str = None,
    log_file_mode: str = "w",
    fmt: str = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    main_process_level: int = logging.INFO,
    other_process_level: int = logging.WARN,
    local_rank: int = -1,
):
    """Shared logic for configuring both root and independent loggers."""
    logger.setLevel(logging.DEBUG)

    # ---- Clear existing handlers ----
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # ---- Console handler ----
    console_handler = logging.StreamHandler()
    console_handler.setLevel(
        main_process_level if local_rank in [-1, 0] else other_process_level
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ---- File handler ----
    if log_file is not None and local_rank in [-1, 0]:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(
            log_file, mode=log_file_mode, encoding="utf-8"
        )
        file_handler.setLevel(main_process_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def setup_root_logger(
    log_file: str = None,
    log_file_mode: str = "w",
    fmt: str = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    main_process_level: int = logging.INFO,
    other_process_level: int = logging.WARN,
    local_rank: int = -1,
):
    """Configure root logger."""
    logger = logging.getLogger()  # root logger
    _configure_logger(
        logger,
        log_file,
        log_file_mode,
        fmt,
        datefmt,
        main_process_level,
        other_process_level,
        local_rank,
    )
    return logger


def get_logger(
    name: str,
    log_file: str = None,
    log_file_mode: str = "w",
    fmt: str = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    main_process_level: int = logging.INFO,
    other_process_level: int = logging.WARN,
    local_rank: int = -1,
):
    """Create a logger independent from root."""
    logger = logging.getLogger(name)
    logger.propagate = False
    _configure_logger(
        logger,
        log_file,
        log_file_mode,
        fmt,
        datefmt,
        main_process_level,
        other_process_level,
        local_rank,
    )
    return logger


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
    config_file = Path(__file__).resolve().parent / "debugpy_config" / "launch.json"
    cfg = read_json(config_file, convert_to_easydict=False)
    cfg["configurations"][-1]["connect"]["port"] = port
    save_path = Path(save_dir) / ".vscode" / "launch.json"
    write_json(cfg, save_path)


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
    def __init__(self, name: list[str]):
        self.name = name
        self.data = [0.0] * len(name)  # initialize to zero
        self.add_times = 0  # track number of times `add` is called

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
        self.add_times += 1
    
    def mean(self):
        self.data = [a / self.add_times for a in self.data]

    def reset(self):
        self.data = [0.0] * len(self.data)
        self.add_times = 0
    
    def to_dict(self):
        return {name: data for name, data in zip(self.name, self.data)}

    def __getitem__(self, key: Union[int, str]):
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