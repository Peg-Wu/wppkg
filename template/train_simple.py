import torch
import pandas as pd
from dataclasses import dataclass, field
from torch.utils.data import random_split
from wppkg import TrainingArguments, Trainer
from transformers import HfArgumentParser, BertTokenizer, BertForSequenceClassification


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
    
    # Create model
    model = BertForSequenceClassification.from_pretrained(model_args.model_name_or_path)

    # Train
    trainer = Trainer(
        args=train_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_func
    )

    trainer.train()


if __name__ == "__main__":
    main()