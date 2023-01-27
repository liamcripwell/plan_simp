import os

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from plan_simp.data.utils import (
    CLASS_LABELS, M_CLASS_LABELS, M_OP_TOKENS, OP_TOKENS, prepend_tokens
)


class BartDataModule(pl.LightningDataModule):

    sep = "#$#"

    def __init__(self, tokenizer, params=None):
        super().__init__()

        self.tokenizer = tokenizer
        self.save_hyperparameters(params)

        self.class_labels = CLASS_LABELS if not self.has_param("multi_split") else M_CLASS_LABELS
        self.op_tokens = OP_TOKENS if not self.has_param("multi_split") else M_OP_TOKENS

    def setup(self, stage):
        # read and prepare input data
        self.train = pd.read_csv(self.hparams.train_file).dropna()
        self.train = self.train.sample(frac=1)[:min(self.hparams.max_samples, len(self.train))] # NOTE: this will actually exclude the last item
        if self.hparams.val_file is not None:
            self.valid = pd.read_csv(self.hparams.val_file).dropna()
        print("All data loaded.")

        # train, validation, test split
        if self.hparams.val_file is None:
            train_span = int(self.hparams.train_split * len(self.train))
            val_span = int((self.hparams.train_split + self.hparams.val_split) * len(self.train))
            self.train, self.valid, self.test = np.split(self.train, [train_span, val_span])
        else:
            self.test = self.train[:16] # arbitrarily have 16 test samples as precaution

        self.train=self.train.reset_index(drop=True)
        self.valid=self.valid.reset_index(drop=True)
        self.test=self.test.reset_index(drop=True)

        if self.has_param("op_col") or self.has_param("reading_lvl"):
            train_seqs = prepend_tokens(self.train, self.hparams.x_col, self.class_labels, self.op_tokens, self.hparams.op_col, self.hparams.reading_lvl)
            valid_seqs = prepend_tokens(self.valid, self.hparams.x_col, self.class_labels, self.op_tokens, self.hparams.op_col, self.hparams.reading_lvl)
            test_seqs = prepend_tokens(self.test, self.hparams.x_col, self.class_labels, self.op_tokens, self.hparams.op_col, self.hparams.reading_lvl)
        else:
            train_seqs = list(self.train[self.hparams.x_col])
            valid_seqs = list(self.valid[self.hparams.x_col])
            test_seqs = list(self.test[self.hparams.x_col])

        if self.has_param("sent_level"):
            # join simple sentences if training sentence-level generative model
            train_labels = [" ".join(eval(y)) for y in self.train[self.hparams.y_col]]
            valid_labels = [" ".join(eval(y)) for y in self.valid[self.hparams.y_col]]
            test_labels = [" ".join(eval(y)) for y in self.test[self.hparams.y_col]]
        else:
            train_labels = list(self.train[self.hparams.y_col])
            valid_labels = list(self.valid[self.hparams.y_col])
            test_labels = list(self.test[self.hparams.y_col])

        train_ = [train_seqs]
        valid_ = [valid_seqs]
        test_ = [test_seqs]
        
        # include standard labels
        train_.append(train_labels)
        valid_.append(valid_labels)
        test_.append(test_labels)

        self.train = list(zip(*train_))
        self.valid = list(zip(*valid_))
        self.test = list(zip(*test_))


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.batch_size, shuffle=True, 
                            num_workers=self.hparams.train_workers, pin_memory=True, 
                            collate_fn=self.prepro_collate)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.hparams.batch_size, num_workers=1, 
                            pin_memory=True, collate_fn=self.prepro_collate)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.batch_size, num_workers=1, 
                            pin_memory=True, collate_fn=self.prepro_collate)
    
    def prepro_collate(self, batch):
        inputs = [x[0] for x in batch]
        labels = [x[-1] for x in batch]
        seqs = self.tokenizer(inputs+labels, max_length=self.hparams.max_length, padding=True, truncation=True, 
                                add_special_tokens=True, return_tensors='pt')

        input_ids = seqs.input_ids[:len(inputs)]
        input_mask = seqs.attention_mask[:len(inputs)]
        labels = seqs.input_ids[len(labels):]

        data = {}

        data["input_ids"] = input_ids
        data["attention_mask"] = input_mask
        data["labels"] = labels

        return data

    def has_param(self, param):
        """Check if param exists and has a non-negative/null value."""
        if param in self.hparams:
            param = self.hparams[param] # set `param` to actual value
            if param is not None:
                if not isinstance(param, bool) or param:
                    return True
        return False

    def get_param(self, param):
        if self.has_param(param):
            return self.hparams[param]
