import os
import math
import torch
import logging
import evaluate
import datasets
import transformers
import pandas as pd
from tqdm.auto import tqdm
from typing import Optional, Union
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from torch.utils.data import random_split
from accelerate import Accelerator, DeepSpeedPlugin
from wppkg import (
    setup_root_logger, 
    print_trainable_parameters, 
    Accumulator
)
from transformers import (
    HfArgumentParser, 
    SchedulerType, 
    get_scheduler, 
    BertTokenizer, 
    BertForSequenceClassification
)

logger = logging.getLogger(__name__)


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
    checkpointing_steps: Union[int, str, None] = field(
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


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="hfl/rbt3"
    )


@dataclass
class DataArguments:
    train_file: str = field(
        default="./ChnSentiCorp_htl_all.csv"
    )


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, train_file: str) -> None:
        super().__init__()
        self.data = pd.read_csv(train_file)
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]
    
    def __len__(self):
        return len(self.data)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    train_args: TrainingArguments

    # Initialize the accelerator.
    accelerator_kwargs = {
        "mixed_precision": train_args.mixed_precision,
        "gradient_accumulation_steps": train_args.gradient_accumulation_steps
    }

    if train_args.with_tracking:
        accelerator_kwargs["log_with"] = train_args.report_to
        accelerator_kwargs["project_dir"] = train_args.output_dir
    
    if train_args.deepspeed is not None:
        accelerator_kwargs["deepspeed_plugin"] = DeepSpeedPlugin(train_args.deepspeed)
    
    accelerator = Accelerator(**accelerator_kwargs)

    # Make one log on every process with the configuration for debugging.
    setup_root_logger(
        log_file=os.path.join(train_args.output_dir, "run.log"), 
        log_file_mode="w",
        local_rank=accelerator.local_process_index
    )
    logger.warning(accelerator.state)
    
    # If passed along, set the training seed now.
    if train_args.seed is not None:
        set_seed(train_args.seed)
    
    # Create datasets
    dataset = CustomDataset(train_file=data_args.train_file)
    train_dataset, eval_dataset = random_split(
        dataset, 
        lengths=[0.8, 0.2], 
        generator=torch.Generator().manual_seed(train_args.seed)
    )

    # Create data_collator
    tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path)
    def collate_func(batch):
        texts, labels = [], []
        for item in batch:
            texts.append(item[0])
            labels.append(item[1])
        inputs = tokenizer(
            texts, 
            max_length=128, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        inputs["labels"] = torch.tensor(labels)
        return inputs

    # Create dataloaders
    common_dataloader_kwargs = {
        "num_workers": train_args.dataloader_num_workers,
        "pin_memory": train_args.dataloader_pin_memory,
        "persistent_workers": train_args.dataloader_persistent_workers,
        "prefetch_factor": train_args.dataloader_prefetch_factor
    }
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=train_args.per_device_train_batch_size,
        collate_fn=collate_func,
        **common_dataloader_kwargs,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        batch_size=train_args.per_device_eval_batch_size,
        collate_fn=collate_func,
        **common_dataloader_kwargs,
    )

    # Create model
    model = BertForSequenceClassification.from_pretrained(model_args.model_name_or_path)
    if accelerator.is_main_process:
        print("*" * 88)
        print_trainable_parameters(model)
        print("*" * 88)

    # Create optimizer
    # Split weights in two groups, one with weight decay and the other not. (Optional)
    no_decay = ["bias", "LayerNorm.weight", "RMSNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": train_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=train_args.learning_rate)

    # NOTE: the training dataloader needs to be prepared before we grab his length below 
    # (cause its length will be shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_args.gradient_accumulation_steps)
    if train_args.max_train_steps is None:
        train_args.max_train_steps = train_args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=train_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=train_args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=train_args.max_train_steps
        if overrode_max_train_steps
        else train_args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        train_args.max_train_steps = train_args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    train_args.num_train_epochs = math.ceil(train_args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = train_args.checkpointing_steps  # integer `n` or "epoch"

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if train_args.with_tracking:
        # If there are parameters beyond the existing train_args, they can also be added to experiment_config.
        experiment_config = vars(train_args) | vars(data_args) | vars(model_args)
        # TensorBoard cannot log Enums, need the raw value
        if isinstance(experiment_config["lr_scheduler_type"], SchedulerType):
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("runs", experiment_config)
    
    # Train!
    total_batch_size = train_args.per_device_train_batch_size * accelerator.num_processes * train_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {train_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {train_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {train_args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(train_args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if train_args.resume_from_checkpoint:
        if train_args.resume_from_checkpoint is not None or train_args.resume_from_checkpoint != "":
            checkpoint_path = train_args.resume_from_checkpoint
            path = os.path.basename(train_args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * train_args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // train_args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    # Inner training loop
    accumulator_train = Accumulator(name=["total_loss"])
    for epoch in range(starting_epoch, train_args.num_train_epochs):
        model.train()
        
        if train_args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), train_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
            
            # We keep track of the loss at each logging_steps
            accumulator_train.add(
                accelerator.reduce(loss.detach().clone(), "mean").item()
            )
            
            # Log training progress
            if completed_steps % train_args.logging_steps == 0:
                accumulator_train.mean()
                log_dict = accumulator_train.to_dict()
                accumulator_train.reset()  # reset accumulator
                extra_log_dict = {
                    "grad_norm": grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm,
                    "lr": lr_scheduler.get_last_lr()[0]
                }
                log_dict = log_dict | extra_log_dict
                log_dict_round = {
                    k: round(v, 6) if k == "lr" else round(v, 4)
                    for k, v in log_dict.items()
                }
                logger.info({"epoch": epoch, "step": completed_steps, **log_dict_round})

                if train_args.with_tracking:
                    accelerator.log(log_dict, step=completed_steps)

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps}"
                    output_dir = os.path.join(train_args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    # Save the model checkpoint.
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        os.path.join(output_dir, "model"), is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )

            if completed_steps >= train_args.max_train_steps:
                break
        
        model.eval()
        losses = []
        metric = evaluate.load("./metrics/accuracy")
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(train_args.per_device_eval_batch_size)))

            # Add other metrics if needed.
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references
            )

        losses = torch.cat(losses)
        eval_loss = torch.mean(losses)
        eval_metric = metric.compute()

        # Log evaluation progress
        eval_log_dict = {
            "eval_loss": eval_loss.item(),
            **eval_metric
        }
        logger.info({"epoch": epoch, **eval_log_dict})

        if train_args.with_tracking:
            accelerator.log(eval_log_dict, step=epoch)

        if train_args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            output_dir = os.path.join(train_args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            # Save the model checkpoint.
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                os.path.join(output_dir, "model"), is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )

    # Save the last model checkpoint.
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        os.path.join(train_args.output_dir, "last_model"), is_main_process=accelerator.is_main_process, save_function=accelerator.save
    )

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()