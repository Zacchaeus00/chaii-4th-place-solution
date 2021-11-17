import numpy as np
import pandas as pd
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import Dataset
from utils import prepare_train_features, prepare_validation_features, convert_answers, jaccard, postprocess_qa_predictions
from transformers import XLMRobertaTokenizerFast, XLMRobertaForQuestionAnswering
import re


class ChaiiDataRetriever:
    def __init__(self, model_name, train_path, max_length, doc_stride, batch_size):
        self.model_name = model_name
        self.train = pd.read_csv(train_path)
        self.train['answers'] = self.train[['answer_start',
                                            'answer_text']].apply(convert_answers, axis=1)
        self.train['id'] = self.train.index
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.batch_size = batch_size
        if 'infoxlm' in model_name:
            self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(
                self.model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.pad_on_right = self.tokenizer.padding_side == "right"

    def prepare_data(self, fold, only_chaii=False, lang=None, removecite=False, splitjoin=False, downext=False):
        print(
            f'fold {fold}, only_chaii {only_chaii}, lang {lang}, removecite {removecite}, splitjoin {splitjoin}')

        def remove_cite(s):
            pattern = r'\[[0-9]*?\]'
            return re.sub(pattern, '', s)

        def spilt_join(s):
            return ' '.join(s.split())

        # only use original source as validation data
        if only_chaii:
            if lang is not None:
                df_train = self.train[(self.train['fold'] != fold) & (self.train['src'] == 'chaii') & (
                    self.train['language'] == lang)].reset_index(drop=True)
                df_valid = self.train[(self.train['fold'] == fold) & (self.train['src'] == 'chaii') & (
                    self.train['language'] == lang)].reset_index(drop=True)
            else:
                df_train = self.train[(self.train['fold'] != fold) & (
                    self.train['src'] == 'chaii')].reset_index(drop=True)
                df_valid = self.train[(self.train['fold'] == fold) & (
                    self.train['src'] == 'chaii')].reset_index(drop=True)
        elif not downext:
            if lang is not None:
                df_train = self.train[(self.train['fold'] != fold) | (self.train['src'] != 'chaii') & (
                    self.train['language'] == lang)].reset_index(drop=True)
                df_valid = self.train[(self.train['fold'] == fold) & (self.train['src'] == 'chaii') & (
                    self.train['language'] == lang)].reset_index(drop=True)
            else:
                df_train = self.train[(self.train['fold'] != fold) | (
                    self.train['src'] != 'chaii')].reset_index(drop=True)
                df_valid = self.train[(self.train['fold'] == fold) & (
                    self.train['src'] == 'chaii')].reset_index(drop=True)
        else:
            df_train = self.train[((self.train['fold'] != fold) & (self.train['src'] == 'chaii')) | (
                (self.train['fold'] == fold) & (self.train['src'] != 'chaii'))].reset_index(drop=True)
            df_valid = self.train[(self.train['fold'] == fold) & (
                self.train['src'] == 'chaii')].reset_index(drop=True)
        if removecite:
            df_train['context'] = df_train['context'].apply(remove_cite)
            df_valid['context'] = df_valid['context'].apply(remove_cite)
        if splitjoin:
            df_train['context'] = df_train['context'].apply(spilt_join)
            df_valid['context'] = df_valid['context'].apply(spilt_join)
        print(f"fold{fold} t/v: {len(df_train)}/{len(df_valid)}")
        self.train_dataset = Dataset.from_pandas(df_train)
        self.valid_dataset = Dataset.from_pandas(df_valid)
        self.tokenized_train_ds = self.train_dataset.map(lambda x: prepare_train_features(x, self.tokenizer, self.max_length, self.doc_stride, self.pad_on_right),
                                                         batched=True,
                                                         remove_columns=self.train_dataset.column_names)
        self.tokenized_valid_ds = self.valid_dataset.map(lambda x: prepare_train_features(x, self.tokenizer, self.max_length, self.doc_stride, self.pad_on_right),
                                                         batched=True,
                                                         remove_columns=self.train_dataset.column_names)
        self.validation_features = self.valid_dataset.map(lambda x: prepare_validation_features(x, self.tokenizer, self.max_length, self.doc_stride, self.pad_on_right),
                                                          batched=True,
                                                          remove_columns=self.valid_dataset.column_names
                                                          )
        self.tokenized_train_ds.set_format(type='torch')
        self.tokenized_valid_ds.set_format(type='torch')

    def train_dataloader(self):
        return DataLoader(self.tokenized_train_ds, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.tokenized_valid_ds, batch_size=self.batch_size, num_workers=8)

    def predict_dataloader(self):
        valid_feats_small = self.validation_features.map(
            lambda example: example, remove_columns=['example_id', 'offset_mapping'])
        valid_feats_small.set_format(type='torch')
        return DataLoader(valid_feats_small, batch_size=self.batch_size, num_workers=8)

    def evaluate_jaccard(self, raw_predictions, n_best_size=20, max_answer_length=30, return_predictions=False):
        '''
        raw_predictions: [start_logits, end_logits]
        shape: (N, L)
        '''
        final_predictions = postprocess_qa_predictions(self.valid_dataset,
                                                       self.validation_features,
                                                       raw_predictions,
                                                       self.tokenizer,
                                                       n_best_size,
                                                       max_answer_length)
        df = pd.DataFrame({'id': final_predictions.keys(),
                           'PredictionString': final_predictions.values()})
        df = df.merge(self.train, on=['id'], how='left')
        df['jaccard'] = df[['answer_text', 'PredictionString']].apply(
            jaccard, axis=1)
        if return_predictions:
            return df.jaccard.mean(), df.groupby('language')['jaccard'].mean(), df
        return df.jaccard.mean(), df.groupby('language')['jaccard'].mean()


class ChaiiDataRetrieverCustom:
    def __init__(self, model_name, train_df, valid_df, max_length, doc_stride, batch_size):
        self.model_name = model_name
        self.train = train_df.reset_index(drop=True)
        self.train['answers'] = self.train[['answer_start',
                                            'answer_text']].apply(convert_answers, axis=1)
        self.train['id'] = self.train.index
        self.valid = valid_df.reset_index(drop=True)
        self.valid['answers'] = self.valid[['answer_start',
                                            'answer_text']].apply(convert_answers, axis=1)
        self.valid['id'] = self.valid.index
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.pad_on_right = self.tokenizer.padding_side == "right"

    def prepare_data(self):
        print(f"t/v: {len(self.train)}/{len(self.valid)}")
        self.train_dataset = Dataset.from_pandas(self.train)
        self.valid_dataset = Dataset.from_pandas(self.valid)
        self.tokenized_train_ds = self.train_dataset.map(lambda x: prepare_train_features(x, self.tokenizer, self.max_length, self.doc_stride, self.pad_on_right),
                                                         batched=True,
                                                         remove_columns=self.train_dataset.column_names)
        self.tokenized_valid_ds = self.valid_dataset.map(lambda x: prepare_train_features(x, self.tokenizer, self.max_length, self.doc_stride, self.pad_on_right),
                                                         batched=True,
                                                         remove_columns=self.valid_dataset.column_names)
        self.validation_features = self.valid_dataset.map(lambda x: prepare_validation_features(x, self.tokenizer, self.max_length, self.doc_stride, self.pad_on_right),
                                                          batched=True,
                                                          remove_columns=self.valid_dataset.column_names
                                                          )
        self.tokenized_train_ds.set_format(type='torch')
        self.tokenized_valid_ds.set_format(type='torch')

    def train_dataloader(self):
        return DataLoader(self.tokenized_train_ds, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.tokenized_valid_ds, batch_size=self.batch_size, num_workers=8)

    def predict_dataloader(self):
        valid_feats_small = self.validation_features.map(
            lambda example: example, remove_columns=['example_id', 'offset_mapping'])
        valid_feats_small.set_format(type='torch')
        return DataLoader(valid_feats_small, batch_size=self.batch_size, num_workers=8)

    def evaluate_jaccard(self, raw_predictions, n_best_size=20, max_answer_length=30):
        '''
        raw_predictions: [start_logits, end_logits]
        shape: (N, L)
        '''
        final_predictions = postprocess_qa_predictions(self.valid_dataset,
                                                       self.validation_features,
                                                       raw_predictions,
                                                       self.tokenizer,
                                                       n_best_size,
                                                       max_answer_length)
        df = pd.DataFrame({'id': final_predictions.keys(),
                           'PredictionString': final_predictions.values()})
        df = df.merge(self.valid, on=['id'], how='left')
        df['jaccard'] = df[['answer_text', 'PredictionString']].apply(
            jaccard, axis=1)
        return df.jaccard.mean(), df.groupby('language')['jaccard'].mean()
