from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoConfig
import os
from utils import seed_everything, convert_answers, prepare_train_features_v2, ChaiiRandomSampler, get_time
import argparse
import pandas as pd
from torch.utils.data import DataLoader, BatchSampler
from engine import Engine
import torch
import transformers
import json
from pathlib import Path
from pprint import pprint
import hashlib
md5 = hashlib.md5()
md5.update('how to use md5 in python hashlib?'.encode('utf-8'))

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
    arg('--dropout', required=True, type=float)
    arg('--downsample', required=True, type=float)
    return parser.parse_args()
args = parse_args()
seed_everything(args.seed)
hyp = {k: v for k, v in args.__dict__.items() if k != 'seed'}
md5.update(json.dumps(hyp).encode('utf-8'))
out_dir = f'./model/{md5.hexdigest()}/'
Path(out_dir).mkdir(parents=True, exist_ok=True)
with open(f'{out_dir}/hyp.json', 'w') as f:
    json.dump(hyp, f, indent=4)
    pprint(hyp)
    print(get_time())
    print('seed', args.seed)

tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
train = pd.read_csv(args.train_path)
train['answers'] = train[['answer_start', 'answer_text']].apply(convert_answers, axis=1)
train_dataset = Dataset.from_pandas(train)
tokenized_train_ds = train_dataset.map(lambda x: prepare_train_features_v2(x, tokenizer, args.max_length, args.doc_stride, tokenizer.padding_side == "right"),
                                                        batched=True, 
                                                        remove_columns=train_dataset.column_names)
tokenized_train_ds.set_format(type='torch')
sampler = BatchSampler(sampler=ChaiiRandomSampler(tokenized_train_ds, downsample=args.downsample), batch_size=args.batch_size, drop_last=False)
dataloader = DataLoader(tokenized_train_ds, batch_sampler=sampler)

cfg = AutoConfig.from_pretrained(args.model_checkpoint)
cfg.hidden_dropout_prob = args.dropout
cfg.attention_probs_dropout_prob = args.dropout
model = AutoModelForQuestionAnswering.from_pretrained(args.model_checkpoint, config=cfg)

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
num_training_steps = args.epochs * len(dataloader)
num_warmup_steps = int(args.warmup_ratio * num_training_steps)
scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps)

engine = Engine(model, optimizer, scheduler, "cuda")
for epoch in range(args.epochs):
    tloss = engine.train(dataloader, accumulation_steps=args.accumulation_steps)
    print(f'ep {epoch}, tloss{tloss}')
engine.save(os.path.join(out_dir, f'{args.seed}.pt'))

