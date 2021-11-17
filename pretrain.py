import pandas as pd
import warnings
import os
import datetime
import json
from pathlib import Path

from transformers import (AutoModel, AutoModelForMaskedLM,
                          AutoTokenizer, PreTrainedTokenizer,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

# https://github.com/huggingface/transformers/blob/master/src/transformers/data/datasets/language_modeling.py
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='')
    arg = parser.add_argument
    arg('--block_size', default=512, type=int)
    arg('--mlm_probability', default=0.15, type=float)
    arg('--model_name', default='../../input/microsoft-infoxlm-large', type=str)
    arg('--output_dir', default='../model/microsoft-infoxlm-large-pretrained', type=str)
    arg('--epochs', default=5, type=int)
    arg('--batch_size', default=4, type=int)
    arg('--gradient_accumulation_steps', default=4, type=int)
    arg('--seed', default=42, type=int)
    arg('--learning_rate', default=3e-5, type=float)
    arg('--warmup_ratio', default=0.1, type=float)
    arg('--stride', default=128, type=int)
    return parser.parse_args()


args = parse_args()
timenow = str(datetime.datetime.now()).split('.')[0]
print(timenow)
print(args)
output_dir = args.output_dir + '_'.join(timenow.split())
Path(output_dir).mkdir(parents=True, exist_ok=True)
with open(f'{output_dir}/hyp.json', 'w') as f:
    d = args.__dict__
    d['time'] = timenow
    json.dump(d, f, indent=4)


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, lines: list, block_size: int):
        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True,
                                   max_length=block_size, return_overflowing_tokens=True, stride=args.stride)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(
            e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


train_data = pd.read_csv(
    '../../input/chaii-hindi-and-tamil-question-answering/train.csv')
test_data = pd.read_csv(
    '../../input/chaii-hindi-and-tamil-question-answering/test.csv')
data = pd.concat([train_data, test_data])

model = AutoModelForMaskedLM.from_pretrained(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    lines=data.context.tolist(),
    block_size=args.block_size)

valid_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    lines=data.context.tolist(),
    block_size=args.block_size)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability)

training_args = TrainingArguments(
    output_dir=output_dir,  # select model path for checkpoint
    overwrite_output_dir=True,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    evaluation_strategy='steps',
    save_total_limit=2,
    eval_steps=50 / (args.batch_size * torch.cuda.device_count()) * 4,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    load_best_model_at_end=True,
    prediction_loss_only=True,
    report_to="none",
    seed=args.seed,
    fp16=True,
    learning_rate=args.learning_rate,
    warmup_ratio=args.warmup_ratio,)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset)

trainer.train()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
