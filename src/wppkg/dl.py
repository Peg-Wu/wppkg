import os
import math
import torch
import logging
import datasets
import multiprocessing as mp

from torch import nn
from pathlib import Path
from tqdm.auto import tqdm
from .logging import get_logger
from collections.abc import Mapping
from accelerate.utils import set_seed
from dataclasses import dataclass, field
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
from .utils import read_json, write_json, Accumulator
from typing import Optional, Union, Literal, List, Any
from transformers.trainer_pt_utils import get_parameter_names
from transformers.data.data_collator import DataCollator, default_data_collator
from transformers import BatchEncoding, PreTrainedModel, SchedulerType, get_scheduler

logger = logging.getLogger(__name__)


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


def generate_default_deepspeed_config(
    config_name: Literal["zero1", "zero2", "zero2_offload", "zero3", "zero3_offload"],
    save_path: str
) -> None:
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


class EarlyStoppingCallback:
    "A callback class that helps with early stopping"
    def __init__(self, min_delta=0, patience=5):
        self.min_delta = min_delta
        self.patience = patience
        self.counter = 0
        self.lowest_loss = float("inf")

    def check_early_stopping(self, eval_loss):
        delta = self.lowest_loss - eval_loss
        if delta >= self.min_delta:
            self.lowest_loss = eval_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


@dataclass
class TrainingArguments:
    seed: int = field(
        default=42, 
        metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written. Defaults to 'trainer_output' if not provided."
        }
    )
    num_train_epochs: int = field(
        default=3, 
        metadata={"help": "Total number of training epochs to perform."}
    )
    max_train_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If > 0: set total number of training steps to perform. Override num_train_epochs."
                "NOTE: `max_train_steps` represents the total number of training steps per GPU/device."
            )
        }
    )
    logging_steps: int = field(
        default=500,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer."
                "NOTE: `logging_steps` represents the total number of logging steps per GPU/device."
            )
        }
    )
    eval_every_n_epochs: int = field(
        default=1,
        metadata={
            "help": "Perform evaluation every n epochs."
        }
    )
    earlystop_patience: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "By default, training will be early-stopped if the evaluation loss fails to improve for `earlystop_patience` consecutive epochs."
                "If `earlystop_patience` is set to `None`, early stopping is disabled."
            )
        }
    )
    per_device_train_batch_size: int = field(
        default=8, 
        metadata={"help": "Batch size per device accelerator core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, 
        metadata={"help": "Batch size per device accelerator core/CPU for evaluation."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    max_grad_norm: float = field(
        default=1.0, 
        metadata={"help": "Max gradient norm."}
    )
    learning_rate: float = field(
        default=5e-5, 
        metadata={"help": "The initial learning rate for AdamW."}
    )
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="linear",
        metadata={"help": "The scheduler type to use."}
    )
    weight_decay: float = field(
        default=0.0, 
        metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    num_warmup_steps: int = field(
        default=0, 
        metadata={
            "help": (
                "Linear warmup over warmup_steps."
                "NOTE: `num_warmup_steps` represents the total number of warmup steps per GPU/device."
            )
        }
    )
    mixed_precision: str = field(
        default="bf16",
        metadata={
            "help": (
                "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). "
                "Bf16 requires PyTorch >= 1.10. and an Nvidia Ampere GPU."
            )
        }
    )
    with_tracking: bool = field(
        default=True,
        metadata={"help": "Whether to enable experiment trackers for logging."}
    )
    report_to: str = field(
        default="all",
        metadata={
            "help": (
                "The integration to report the results and logs to. Supported platforms are "
                "'tensorboard', 'wandb', 'comet_ml' and 'clearml'. Use 'all' (default) to report to all integrations. "
                "Only applicable when `--with_tracking` is passed."
            )
        }
    )
    checkpointing_steps: Optional[Union[int, str]]= field(
        default=None,
        metadata={"help": "Whether thr variousstates should be saved at the end of every n steps, or `epoch` for each epoch."}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "If the training should continue from a checkpoint folder."}
    )
    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether or not to pin memory for DataLoader."}
    )
    dataloader_persistent_workers: bool = field(
        default=False,
        metadata={
            "help": "If True, the data loader will not shut down the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive. Can potentially speed up training, but will increase RAM usage."
        }
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        }
    )
    dataloader_prefetch_factor: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of batches loaded in advance by each worker. "
                "2 means there will be a total of 2 * num_workers batches prefetched across all workers. "
            )
        }
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the DeepSpeed config file."
        }
    )

    def __post_init__(self):
        # Set default output_dir if not provided
        if self.output_dir is None:
            self.output_dir = "trainer_output"
            logger.info(
                "No output directory specified, defaulting to 'trainer_output'. "
                "To change this behavior, specify --output_dir when creating TrainingArguments."
            )
        if self.dataloader_num_workers == 0 and self.dataloader_prefetch_factor is not None:
            raise ValueError(
                "--dataloader_prefetch_factor can only be set when data is loaded in a different process, i.e."
                " when --dataloader_num_workers > 1."
            )
        if self.mixed_precision not in ["no", "fp16", "bf16", "fp8"]:
            raise ValueError(
                "--mixed_precision can only be 'no', 'fp16', 'bf16' or 'fp8'."
            )
        if self.checkpointing_steps is not None and self.checkpointing_steps.isdigit():
            self.checkpointing_steps = int(self.checkpointing_steps)


class Trainer:
    def __init__(
        self,
        args: TrainingArguments,
        model: Union[PreTrainedModel, nn.Module],
        train_dataset: Union[Dataset, datasets.Dataset],
        eval_dataset: Optional[Union[Dataset, datasets.Dataset]] = None,
        data_collator: Optional[DataCollator] = None,
    ):
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator if data_collator is not None else default_data_collator

        self.accelerator = self._init_accelerator()
        self.logger = self._init_logger()
        self._report_accelerator_state()
        self._report_model_trainable_parameters()
        set_seed(self.args.seed)

        self.train_dataloader = self.get_train_dataloader()
        self.eval_dataloader = self.get_eval_dataloader()
        self.optimizer = self.create_optimizer()

        # Scheduler and math around the number of training steps.
        self.overrode_max_train_steps = False
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * self.num_update_steps_per_epoch
            self.overrode_max_train_steps = True

        self.lr_scheduler = self.create_scheduler()

        # Prepare everything with `accelerator`.
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.overrode_max_train_steps:
            self.args.max_train_steps = self.args.num_train_epochs * self.num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / self.num_update_steps_per_epoch)

        # EarlyStop CallBack
        self.earlystop_callback = (
            EarlyStoppingCallback(patience=self.args.earlystop_patience)
            if self.args.earlystop_patience is not None
            else None
        )

        self._init_trackers()

    def _init_accelerator(self) -> Accelerator:
        # Initialize the accelerator.
        accelerator_kwargs = {
            "mixed_precision": self.args.mixed_precision,
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps
        }

        if self.args.with_tracking:
            accelerator_kwargs["log_with"] = self.args.report_to
            accelerator_kwargs["project_dir"] = self.args.output_dir
        
        if self.args.deepspeed is not None:
            accelerator_kwargs["deepspeed_plugin"] = DeepSpeedPlugin(self.args.deepspeed)
        
        return Accelerator(**accelerator_kwargs)

    def _init_logger(self) -> logging.Logger:
        # Create an independent logger for the Trainer.
        return get_logger(
            name="wppkg.Trainer",
            log_file=os.path.join(self.args.output_dir, "run.log"),
            log_file_mode="w",
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            main_process_level=logging.INFO,
            other_process_level=logging.WARN,
            local_rank=self.accelerator.local_process_index
        )
    
    def _init_trackers(self):
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.args.with_tracking:
            experiment_config = vars(self.args)
            # TensorBoard cannot log Enums, need the raw value
            if isinstance(experiment_config["lr_scheduler_type"], SchedulerType):
                experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            self.accelerator.init_trackers("runs", experiment_config)
    
    def _report_accelerator_state(self):
        """Log the current state of the Accelerator (device, process info, distributed setup, etc.)."""
        self.logger.warning(
            "Accelerator state:\n%s", self.accelerator.state
        )
    
    def _report_model_trainable_parameters(self):
        trainable_params, all_param = get_nb_trainable_parameters(self.model)
        self.logger.info(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    
    def get_decay_parameter_names(self, model) -> list[str]:
        """
        Get all parameter names that weight decay will be applied to.

        This function filters out parameters in two ways:
        1. By layer type (instances of layers specified in ALL_LAYERNORM_LAYERS)
        2. By parameter name patterns (containing 'bias', or variation of 'norm')
        """
        forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
        decay_parameters = get_parameter_names(model, [nn.LayerNorm], forbidden_name_patterns)
        return decay_parameters

    def create_optimizer(self):
        decay_parameters = self.get_decay_parameter_names(self.model)
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
    
    def create_scheduler(self):
        return get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.num_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.args.max_train_steps
            if self.overrode_max_train_steps
            else self.args.max_train_steps * self.accelerator.num_processes,
        )

    def get_train_dataloader(self):
        common_dataloader_kwargs = {
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "prefetch_factor": self.args.dataloader_prefetch_factor
        }
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            **common_dataloader_kwargs,
        )
    
    def get_eval_dataloader(self):
        if self.eval_dataset is None:
            return 
        else:
            common_dataloader_kwargs = {
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
                "prefetch_factor": self.args.dataloader_prefetch_factor
            }
            return DataLoader(
                self.eval_dataset,
                shuffle=False,
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=self.data_collator,
                **common_dataloader_kwargs,
            )
    
    def _save(self, save_dir: str):
        # Save the model checkpoint.
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            save_dir, 
            is_main_process=self.accelerator.is_main_process, 
            save_function=self.accelerator.save
        )
    
    def train(self):
        # Train!
        total_batch_size = self.args.per_device_train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_dataset)}")
        self.logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        self.logger.info("*****************************")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.args.max_train_steps), disable=not self.accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0

        # Potentially load in the weights and states from a previous save
        if self.args.resume_from_checkpoint:
            if self.args.resume_from_checkpoint is not None or self.args.resume_from_checkpoint != "":
                checkpoint_path = self.args.resume_from_checkpoint
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
                checkpoint_path = path
                path = os.path.basename(checkpoint_path)

            self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
            self.accelerator.load_state(checkpoint_path)
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
                completed_steps = starting_epoch * self.num_update_steps_per_epoch
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = int(training_difference.replace("step_", "")) * self.args.gradient_accumulation_steps
                starting_epoch = resume_step // len(self.train_dataloader)
                completed_steps = resume_step // self.args.gradient_accumulation_steps
                resume_step -= starting_epoch * len(self.train_dataloader)

        # update the progress_bar if load from checkpoint
        progress_bar.update(completed_steps)

        # NOTE: Inner training loop
        # TODO: Add other losses if needed.
        accumulator_train = Accumulator(name=["train_loss"])
        for epoch in range(starting_epoch, self.args.num_train_epochs):
            self.model.train()
        
            if self.args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
                # We skip the first `n` batches in the dataloader when resuming from a checkpoint
                active_dataloader = self.accelerator.skip_first_batches(self.train_dataloader, resume_step)
            else:
                active_dataloader = self.train_dataloader
            for step, batch in enumerate(active_dataloader):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                
                # We keep track of the loss at each logging_steps
                accumulator_train.add(
                    self.accelerator.reduce(loss.detach().clone(), "mean").item()
                )
                
                # Log training progress
                if completed_steps % self.args.logging_steps == 0:
                    accumulator_train.mean()
                    log_dict = accumulator_train.to_dict()
                    accumulator_train.reset()  # reset accumulator
                    extra_log_dict = {
                        "grad_norm": grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm,
                        "lr": self.lr_scheduler.get_last_lr()[0]
                    }
                    log_dict = log_dict | extra_log_dict
                    log_dict_round = {
                        k: round(v, 6) if k == "lr" else round(v, 4)
                        for k, v in log_dict.items()
                    }
                    self.logger.info({"epoch": epoch, "step": completed_steps, **log_dict_round})

                    if self.args.with_tracking:
                        self.accelerator.log(log_dict, step=completed_steps)

                if isinstance(self.args.checkpointing_steps, int):
                    if completed_steps % self.args.checkpointing_steps == 0 and self.accelerator.sync_gradients:
                        output_dir = f"step_{completed_steps}"
                        output_dir = os.path.join(self.args.output_dir, output_dir)
                        self.accelerator.save_state(output_dir)
                        # Save the model checkpoint et al.
                        self._save(os.path.join(output_dir, "model"))

                if completed_steps >= self.args.max_train_steps:
                    break
            
            # NOTE: Evaluation will be performed at the end of each epoch. (or every `eval_every_n_epochs`)
            if self.eval_dataloader is not None and (epoch + 1) % self.args.eval_every_n_epochs == 0:
                eval_log_dict = self.evaluate()

                # Log evaluation progress
                self.logger.info({"epoch": epoch, **eval_log_dict})
                if self.args.with_tracking:
                    self.accelerator.log(eval_log_dict, step=epoch)

            if self.args.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                output_dir = os.path.join(self.args.output_dir, output_dir)
                self.accelerator.save_state(output_dir)
                # Save the model checkpoint et al.
                self._save(os.path.join(output_dir, "model"))
            
            # EarlyStop: check if we should stop the training on any processes
            if self.earlystop_callback is not None:
                if self.earlystop_callback.check_early_stopping(eval_log_dict["eval_loss"]):
                    self.accelerator.set_trigger()
                # If so, we break the loop
                if self.accelerator.check_trigger():
                    self.logger.info(f"Model has not improved for {self.args.earlystop_patience} epochs, so we halt the training session.")
                    break

        # Save the last model checkpoint.
        self._save(os.path.join(self.args.output_dir, "last_model"))
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
        self.logger.info("Training exited successfully.")
    
    def evaluate(self):
        self.model.eval()
        losses = []
        for step, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                outputs = self.model(**batch)
            
            # TODO: Add other losses if needed.
            loss = outputs.loss
            losses.append(self.accelerator.gather_for_metrics(loss.repeat(self.args.per_device_eval_batch_size)))

            # TODO: Add other metrics if needed.
            # predictions = outputs.logits.argmax(dim=-1)
            # predictions, references = self.accelerator.gather_for_metrics((predictions, batch["labels"]))
            # metric.add_batch(
            #     predictions=predictions,
            #     references=references,
            # )

        eval_loss = torch.mean(torch.cat(losses))
        # eval_metric = metric.compute()
        return {
            "eval_loss": eval_loss.item(),
            # **eval_metrics
        }
