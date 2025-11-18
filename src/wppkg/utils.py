import os
import json
import logging
from typing import Union

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