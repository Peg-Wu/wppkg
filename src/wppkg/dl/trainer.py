import os
import math
import torch
import logging
import datasets
from torch import nn
from tqdm.auto import tqdm
from .utils import Accumulator
from ..logging import get_logger
from typing import Optional, Union
from accelerate.utils import set_seed
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
from transformers.trainer_pt_utils import get_parameter_names
from transformers import PreTrainedModel, SchedulerType, get_scheduler
from transformers.data.data_collator import DataCollator, default_data_collator

logger = logging.getLogger(__name__)


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
                "By default, training will be early-stopped if the evaluation loss fails to improve for `earlystop_patience` consecutive evaluations."
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
    num_warmup_ratio: Optional[float] = field(
        default=None, metadata={
            "help": "Warmup ratio of total optimization steps, in the range [0, 1]. If specified, it will override `num_warmup_steps`."
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
        metadata={"help": "When to save checkpoints: int = every N steps, 'epoch' = every epoch, 'epoch-k' = every k epochs."}
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
        if self.num_warmup_ratio is not None:
            assert 0 <= self.num_warmup_ratio <= 1, "`num_warmup_ratio` must be in the range [0, 1]."


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
        # total training steps
        num_training_steps = (
            self.args.max_train_steps
            if self.overrode_max_train_steps
            else self.args.max_train_steps * self.accelerator.num_processes
        )

        # total warmup steps
        if self.args.num_warmup_ratio is not None:
            num_warmup_steps = int(self.args.num_warmup_ratio * num_training_steps)
        else:
            num_warmup_steps = int(self.args.num_warmup_steps * self.accelerator.num_processes)

        return get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
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
        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(save_dir, "training_args.bin"))
    
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

            self.logger.info(f"Resumed from checkpoint: {checkpoint_path}")
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
                
                # EarlyStop: check if we should stop the training on any processes
                if self.earlystop_callback is not None:
                    if self.earlystop_callback.check_early_stopping(eval_log_dict["eval_loss"]):
                        self.accelerator.set_trigger()
                    # If so, we break the loop
                    if self.accelerator.check_trigger():
                        self.logger.info(f"Model has not improved for {self.args.earlystop_patience} evaluations, so we halt the training session.")
                        break

            # NOTE: Allow checkpointing_steps to be in the format "epoch-<number>", meaning a checkpoint is saved every <number> epochs.
            if isinstance(self.args.checkpointing_steps, str):
                checkpointing_every_n_epochs = (
                    1 
                    if self.args.checkpointing_steps == "epoch" 
                    else int(self.args.checkpointing_steps.split("-")[-1])
                )

                if (epoch + 1) % checkpointing_every_n_epochs == 0:
                    output_dir = f"epoch_{epoch}"
                    output_dir = os.path.join(self.args.output_dir, output_dir)
                    self.accelerator.save_state(output_dir)
                    # Save the model checkpoint et al.
                    self._save(os.path.join(output_dir, "model"))

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