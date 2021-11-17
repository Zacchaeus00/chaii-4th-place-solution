import numpy as np
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import transformers
from transformers import AutoModelForQuestionAnswering, AutoConfig
from data import ChaiiDataRetriever
from madgrad import MADGRAD
from engine import Engine
from utils import seed_everything, log_scores, log_hyp
import datetime
from pprint import pprint
import logging
import argparse
from utils import get_time
from pathlib import Path
import json
import uuid


def parse_args():
    parser = argparse.ArgumentParser(description='')
    arg = parser.add_argument
    arg('--model_checkpoint', required=True, type=str)
    arg('--train_path', required=True, type=str)
    arg('--max_length', required=True, type=int)
    arg('--doc_stride', required=True, type=int)
    arg('--epochs', required=True, type=int)
    arg('--batch_size', required=True, type=int)
    arg('--accumulation_steps', required=True, type=int)
    arg('--lr', required=True, type=float)
    arg('--optimizer', required=True, type=str)
    arg('--weight_decay', required=True, type=float)
    arg('--scheduler', required=True, type=str)
    arg('--warmup_ratio', required=True, type=float)
    arg('--dropout', required=True, type=float)
    arg('--eval_steps', required=True, type=int)
    arg('--metric', required=True, type=str)
    arg('--downext', dest='downext', action='store_true')
    arg('--seed', required=True, type=int)
    return parser.parse_args()


timenow = get_time()
out_dir = f'./model/{timenow}-{uuid.uuid1()}/'
args = parse_args()
seed_everything(args.seed)
Path(out_dir).mkdir(parents=True, exist_ok=True)
with open(f'{out_dir}/hyp.json', 'w') as f:
    d = args.__dict__
    d['time'] = timenow
    json.dump(d, f, indent=4)
    pprint(d)

data_retriever = ChaiiDataRetriever(
    args.model_checkpoint, args.train_path, args.max_length, args.doc_stride, args.batch_size)
folds = 5
oof_scores = np.zeros(folds)
for fold in range(folds):
    print("fold", fold)
    data_retriever.prepare_data(fold, downext=args.downext)
    train_dataloader = data_retriever.train_dataloader()
    val_dataloader = data_retriever.val_dataloader()
    predict_dataloader = data_retriever.predict_dataloader()
    cfg = AutoConfig.from_pretrained(args.model_checkpoint)
    cfg.hidden_dropout_prob = args.dropout
    cfg.attention_probs_dropout_prob = args.dropout
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_checkpoint, config=cfg)

    num_training_steps = args.epochs * len(train_dataloader)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    if args.optimizer == 'madgrad':
        optimizer = MADGRAD(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
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
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.scheduler == 'cosann':
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps)
    elif args.scheduler == 'linann':
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps)
    else:
        scheduler = None

    engine = Engine(model, optimizer, scheduler, 'cuda')
    raw_predictions = engine.predict(predict_dataloader)
    best_score, lang_scores, df = data_retriever.evaluate_jaccard(
        raw_predictions, return_predictions=True)
    nonzero_jaccard_per = len(df[df['jaccard'] != 0]) / len(df)
    print(f'initial mean jaccard {best_score}')
    print(f'initial nonzero jaccard percentage {nonzero_jaccard_per}')
    best_metric = best_score if args.metric == 'mean_jaccard' else nonzero_jaccard_per
    print(f'using metric: {args.metric}')
    for epoch in range(args.epochs):
        best_metric = engine.train_evaluate(train_dataloader,
                                            predict_dataloader,
                                            data_retriever,
                                            args.eval_steps,
                                            best_metric,
                                            out_dir + f'fold{fold}.pt',
                                            args.metric,
                                            accumulation_steps=args.accumulation_steps)

    print(f'fold {fold} best {args.metric} {best_metric}')
    oof_scores[fold] = best_metric
    # torch.save(model.state_dict(), out_dir+f'fold{fold}_last.pt')
print(f'{folds} fold cv {args.metric}: {oof_scores.mean()}')
log_scores(out_dir, oof_scores)

# evaluate
data_retriever = ChaiiDataRetriever(
    args.model_checkpoint, args.train_path, args.max_length, args.doc_stride, args.batch_size)
folds = 5
hindi_scores = []
tamil_scores = []
scores = []
all_df = pd.DataFrame()
for fold in range(folds):
    print("fold", fold)
    data_retriever.prepare_data(fold, only_chaii=True)
    predict_dataloader = data_retriever.predict_dataloader()
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_checkpoint)
    model.load_state_dict(torch.load(os.path.join(out_dir, f'fold{fold}.pt')))

    engine = Engine(model, None, None, 'cuda')
    raw_predictions = engine.predict(predict_dataloader)
    score, lang_scores, df = data_retriever.evaluate_jaccard(
        raw_predictions, return_predictions=True)
    all_df = pd.concat([all_df, df], axis=0)
    hindi_scores.append(lang_scores['hindi'])
    tamil_scores.append(lang_scores['tamil'])
    scores.append(score)
    print(score)
    print(lang_scores)
logging.basicConfig(filename=os.path.join(
    out_dir, 'evaluate.log'), level=logging.DEBUG)
logging.info('hindi mean: {}'.format(np.mean(hindi_scores)))
logging.info('tamil mean: {}'.format(np.mean(tamil_scores)))
logging.info('macro mean: {}'.format(
    (np.mean(hindi_scores) + np.mean(tamil_scores)) / 2))
logging.info('micro mean: {}'.format(np.mean(scores)))
all_df.to_csv(os.path.join(out_dir, 'oof_predictions.csv'), index=False)
