import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.utils import resample
from torch.utils.data import DataLoader

from plan_simp.data.utils import CLASS_LABELS, M_CLASS_LABELS, READING_LVLS, convert_labels, prepare_batch_context


def upsample(df, y_col="label", seed=1, keep_all=True):
    """Upsample dataset so that minority classes have the same size as the majority class."""
    maj = df[y_col].value_counts().idxmax()
    df_maj = df[df[y_col] == maj]
    non_maj = [c for c in df[y_col].unique() if c != maj]

    upsamps = []
    for c in non_maj:
        df_c = df[df[y_col] == c]
        if keep_all:
            df_upsamp = resample(df_c, replace=True, n_samples=len(df_maj)-len(df_c), random_state=seed)
            df_upsamp = pd.concat([df_c, df_upsamp])
        else:
            df_upsamp = resample(df_c, replace=True, n_samples=len(df_maj), random_state=seed)
        upsamps.append(df_upsamp)

    df_upsampled = pd.concat([df_maj] + upsamps).sample(frac=1)

    print(f"Before upsampling:\n{df[y_col].value_counts()}")
    print(f"After upsampling:\n{df_upsampled[y_col].value_counts()}")

    return df_upsampled

def append_tokens(df, x_col, tokens):
    """Append specified control tokens to text and return results as List."""
    seqs = [f"{' '.join([READING_LVLS[row[t]] for t in tokens])} {row[x_col]}" for _, row in df.iterrows()]
    return seqs


class RobertaDataModule(pl.LightningDataModule):

    sep = "#$#"

    def __init__(self, tokenizer, params=None):
        super().__init__()

        self.tokenizer = tokenizer
        self.save_hyperparameters(params)

        self.class_labels = CLASS_LABELS if not self.has_param("multi_split") else M_CLASS_LABELS

        # NOTE: hacky way to still use older models without `src_lvl`
        if not self.has_param("src_lvl"):
            self.hparams.src_lvl = None
        if not self.has_param("reading_lvl"):
            self.hparams.reading_lvl = None
        tok_cols = [self.hparams.src_lvl, self.hparams.reading_lvl]
        self.ctrl_tokens = [t for t in tok_cols if t is not None]

    def setup(self, stage):
        # read and prepare input data
        self.train = pd.read_csv(self.hparams.train_file).dropna()
        self.train = self.train.sample(frac=1)[:min(self.hparams.max_samples, len(self.train))] # NOTE: this will actually exclude the last item
        if self.has_param("val_file"):
            self.valid = pd.read_csv(self.hparams.val_file).dropna()
        print("All data loaded.")

        # train, validation, test split
        if self.hparams.val_file is None:
            train_span = int(self.hparams.train_split * len(self.train))
            val_span = int((self.hparams.train_split + self.hparams.val_split) * len(self.train))
            self.train, self.valid, self.test = np.split(self.train, [train_span, val_span])
        else:
            self.test = self.train[:16] # arbitrarily have 16 test samples as precaution

        # upsample rarer class if desired
        if self.hparams.upsample_classes:
            self.train = upsample(self.train, y_col=self.hparams.y_col, seed=1, keep_all=True)

        if self.ctrl_tokens == []:
            train_seqs = list(self.train[self.hparams.x_col])
            valid_seqs = list(self.valid[self.hparams.x_col])
            test_seqs = list(self.test[self.hparams.x_col])
        else:
            train_seqs = append_tokens(self.train, self.hparams.x_col, self.ctrl_tokens)
            valid_seqs = append_tokens(self.valid, self.hparams.x_col, self.ctrl_tokens)
            test_seqs = append_tokens(self.test, self.hparams.x_col, self.ctrl_tokens)

        if self.hparams.add_context:
            # include context ids in batches
            train_ = [train_seqs, self.prepare_context_ids(self.train, self.hparams.context_doc_id)]
            valid_ = [valid_seqs, self.prepare_context_ids(self.valid, self.hparams.context_doc_id)]
            test_ = [test_seqs, self.prepare_context_ids(self.test, self.hparams.context_doc_id)]

            # include simple document ids in case of dynamic context option
            if self.has_param("simple_context_dir"):
                train_.append(self.prepare_context_ids(
                    self.train, self.hparams.simple_context_doc_id, "simp_sent_id"))
                valid_.append(self.prepare_context_ids(
                    self.valid, self.hparams.simple_context_doc_id, "simp_sent_id"))
                test_.append(self.prepare_context_ids(
                    self.test, self.hparams.simple_context_doc_id, "simp_sent_id"))

            # include document position information if required
            if self.has_param("doc_pos_embeds"):
                train_.append(list(self.train[["doc_pos", "doc_len"]].itertuples(index=False, name=None)))
                valid_.append(list(self.valid[["doc_pos", "doc_len"]].itertuples(index=False, name=None)))
                test_.append(list(self.test[["doc_pos", "doc_len"]].itertuples(index=False, name=None)))

            # include labels as last item in batch
            train_.append(list(self.train[self.hparams.y_col]))
            valid_.append(list(self.valid[self.hparams.y_col]))
            test_.append(list(self.test[self.hparams.y_col]))

            self.train = list(zip(*train_))
            self.valid = list(zip(*valid_)) 
            self.test = list(zip(*test_))
        else:
            self.train = list(zip(train_seqs, list(self.train[self.hparams.y_col])))
            self.valid = list(zip(valid_seqs, list(self.valid[self.hparams.y_col])))
            self.test = list(zip(test_seqs, list(self.test[self.hparams.y_col])))

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

    def prepare_context_ids(self, df, doc_id_col="c_id", sent_id_col="sent_id"):
        """Prepare context as lookup key."""
        if sent_id_col is None:
            # NOTE: this will be the case for simple context during dynamic generation
            # here, we want to use the entire cached simple context, therefore we use -1 as the id
            return [f"{row[doc_id_col]}{self.sep}{-1}" for _, row in df.iterrows()]
        return [f"{row[doc_id_col]}{self.sep}{row[sent_id_col]}" for _, row in df.iterrows()]

    def prepro_collate(self, batch):
        inputs = [x[0] for x in batch]

        # interpret labels if required
        labels = None
        if self.batch_has_labels(batch):
            labels = convert_labels([x[-1] for x in batch], self.class_labels)

            # binary keep/delete classification option
            if self.has_param("binary_clf"):
                labels = [int(y < 4) for y in labels]

        data = self.tokenizer(inputs, max_length=128, padding=True, truncation=True, 
                                add_special_tokens=True, return_tensors='pt')

        # prepare context representations
        if self.hparams.add_context:
            context_data = prepare_batch_context(self, batch, labels=labels)
            data = {**data, **context_data}

        if labels is not None:
            if self.has_param("regression"):
                data["labels"] = torch.tensor(labels).float()
            else:
                data["labels"] = torch.tensor(labels)

        return data
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # only send tensors to device
        batch = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return batch

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

    def batch_has_labels(self, batch):
        # deduce whether batches contain labels
        num_batch_items = len(batch[0])
        num_expected = 1 + sum([self.has_param(p) for p in 
                            ["add_context", "simple_context_dir", "doc_pos_embeds"]])
        return num_batch_items > num_expected
