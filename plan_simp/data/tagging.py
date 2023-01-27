import re

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from plan_simp.data.utils import CLASS_LABELS, M_CLASS_LABELS, OP_TOKENS, M_OP_TOKENS, READING_LVLS, convert_labels

def get_ctrl_tokens(df, tokens):
    return [" ".join([READING_LVLS[row[t]] for t in tokens]) for _, row in df.iterrows()]


class TaggerDataModule(pl.LightningDataModule):

    def __init__(self, tokenizer=None, params=None):
        super().__init__()

        self.tokenizer = tokenizer

        self.save_hyperparameters(params)

        self.class_labels = CLASS_LABELS if not self.has_param("multi_split") else M_CLASS_LABELS
        self.op_tokens = OP_TOKENS if not self.has_param("multi_split") else M_OP_TOKENS

        if not self.has_param("src_lvl"):
            self.hparams.src_lvl = None
        if not self.has_param("reading_lvl"):
            self.hparams.reading_lvl = None
        self.ctrl_tokens = [t for t in [self.hparams.reading_lvl] if t is not None]

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

        train_seqs = list(self.train[self.hparams.x_col])
        valid_seqs = list(self.valid[self.hparams.x_col])
        test_seqs = list(self.test[self.hparams.x_col])

        if self.ctrl_tokens == []:
            self.train = list(zip(train_seqs, list(self.train[self.hparams.y_col])))
            self.valid = list(zip(valid_seqs, list(self.valid[self.hparams.y_col])))
            self.test = list(zip(test_seqs, list(self.test[self.hparams.y_col])))
        else:
            self.train = list(zip(train_seqs, get_ctrl_tokens(self.train, self.ctrl_tokens), list(self.train[self.hparams.y_col])))
            self.valid = list(zip(valid_seqs, get_ctrl_tokens(self.valid, self.ctrl_tokens), list(self.valid[self.hparams.y_col])))
            self.test = list(zip(test_seqs, get_ctrl_tokens(self.test, self.ctrl_tokens), list(self.test[self.hparams.y_col])))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.batch_size, shuffle=True, 
                            num_workers=self.hparams.train_workers, pin_memory=True, 
                            collate_fn=self.prepro_collate)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.hparams.batch_size, 
                            num_workers=self.hparams.val_workers, pin_memory=True, 
                            collate_fn=self.prepro_collate)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.batch_size, num_workers=1, 
                            pin_memory=False, collate_fn=self.prepro_collate)

    def labels_to_seq(self, labels):
        return " ".join([self.op_tokens[i] for i in labels])

    def prepro_collate(self, batch):
        doc_ids = [x[0] for x in batch]
        labels = None

        if len(batch[0]) > 1:
            # if only two elements, assume second is labels if no ctrl tokens
            if not (len(batch[0]) == 2 and self.ctrl_tokens != []):
                labels = [eval(x[-1]) for x in batch]
                labels = [convert_labels(ls_i, self.class_labels) for ls_i in labels]

                if not self.has_param("use_decoder"):
                    labels = [torch.tensor(ls) for ls in labels]
                else:
                    # tokenize operation labels if using seq2seq approach
                    label_seqs = [self.labels_to_seq(ls) for ls in labels]
                    labels = self.tokenizer(label_seqs, return_tensors="pt", padding=True)["input_ids"]

        data = {}

        if self.has_param("use_bart"):
            inputs = doc_ids

            # add ctrl tokens to beginning of document
            sents = [[xi.strip() for xi in x.split("<SEP>")] for x in inputs]
            num_sents = [torch.tensor(len(x)) for x in sents]
            input_seqs = ["</s> <s>".join(x) for x in sents]
            if self.ctrl_tokens != []:
                ctrl_toks = [x[1] for x in batch]
                input_seqs = [f"{ctrl_toks[i]} {t}" for i, t in enumerate(input_seqs)]
                ctrl_toks = [f"{self.tokenizer.cls_token} {tok}" for tok in ctrl_toks]

            # tokenize actual full document content
            data = self.tokenizer(input_seqs, max_length=1024, padding=True, truncation=True, 
                                    add_special_tokens=True, return_tensors='pt')
            data["num_sents"] = torch.stack(num_sents)

            # NOTE: we keep `ctrl_tok_ids` in the batch here specifically for the generated sequence clipping
            # by having both the CLS and reading level tokens, we can simply reuse the code from the AR option
            if self.ctrl_tokens != []:
                ctrl_toks = self.tokenizer(ctrl_toks, add_special_tokens=False, return_tensors="pt")["input_ids"]
                data["ctrl_tok_ids"] = ctrl_toks
        else:
            # load sentence embeddings for document
            inputs = []
            max_batch_len = 0
            for doc_id in doc_ids:
                seq = torch.load(f"{self.hparams.embed_dir}/{doc_id}_z.pt")
                inputs.append(seq)
                if len(seq) > max_batch_len:
                    max_batch_len = len(seq)

            masks = []
            for i, x in enumerate(inputs):
                # input padding
                masks.append(torch.cat([torch.ones(len(x)), torch.zeros(max_batch_len-len(x))]))
                inputs[i] = torch.cat([x, torch.zeros(768).repeat(max_batch_len-len(x), 1)])
            data["inputs_embeds"] = torch.stack(inputs)
            data["attention_mask"] = torch.stack(masks)

            # tokenize non-sentence tokens
            # NOTE: these are not inserted into input here because we need access to the model
            # to generate embedding representations.
            ctrl_toks = []
            for i, x in enumerate(inputs):
                # tokenize all ctrl tokens (including CLS)
                toks = self.tokenizer.cls_token
                if self.ctrl_tokens != []:
                    toks = f"{toks} {batch[i][1]}"
                toks = self.tokenizer(toks, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                ctrl_toks.append(toks)
            data["ctrl_tok_ids"] = torch.stack(ctrl_toks)

        if labels is not None:
            # label padding (already done during tokenization if using seq2seq)
            if not self.has_param("use_decoder"):
                for i, y in enumerate(labels):
                    labels[i] = torch.cat([y, torch.zeros(max_batch_len-len(y))-100]).to(dtype=torch.long)
                labels = torch.stack(labels)

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
