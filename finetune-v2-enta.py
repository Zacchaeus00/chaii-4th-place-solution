from datasets import Dataset
from numpy.core.numeric import _rollaxis_dispatcher
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoConfig, TrainingArguments, Trainer, default_data_collator
import os
from utils import seed_everything, read_squad_enta, prepare_train_features, get_time
from pprint import pprint
import datetime
import argparse
import json
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def parse_args():
    parser = argparse.ArgumentParser(description='')
    arg = parser.add_argument
    arg('--model_checkpoint', required=True, type=str)
    arg('--train_path', required=True, type=str, default='/gpfsnyu/scratch/yw3642/chaii/input/squad2/train-v2.0.json')
    arg('--max_length', required=True, type=int)
    arg('--doc_stride', required=True, type=int)
    arg('--epochs', required=True, type=int)
    arg('--batch_size', required=True, type=int)
    arg('--accumulation_steps', required=True, type=int)
    arg('--lr', required=True, type=float)
    arg('--weight_decay', required=True, type=float)
    arg('--warmup_ratio', required=True, type=float)
    arg('--seed', required=True, type=int)
    return parser.parse_args()
args = parse_args()
seed_everything(args.seed)
train_dataset = Dataset.from_dict(read_squad_enta(args.train_path))
tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
tokenized_train_ds = train_dataset.map(lambda x: prepare_train_features(x, tokenizer, args.max_length, args.doc_stride, tokenizer.padding_side == "right"), batched=True, remove_columns=train_dataset.column_names)
model = AutoModelForQuestionAnswering.from_pretrained(args.model_checkpoint)

timenow = get_time()
out_dir = Path(f'./model/{timenow}/')
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir/'hyp.json', 'w') as f:
    d = args.__dict__
    d['time'] = timenow
    json.dump(d, f, indent=4)
    pprint(d)
args = TrainingArguments(
    out_dir,
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=args.lr,
    warmup_ratio=args.warmup_ratio,
    gradient_accumulation_steps=args.accumulation_steps,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    fp16=True,
    report_to='none',
    dataloader_num_workers=4
)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_train_ds,
    tokenizer=tokenizer,
)
trainer.train()
print(datetime.datetime.now())
