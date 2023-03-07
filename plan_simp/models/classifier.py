import argparse

import numpy as np
import torch
from tqdm import tqdm
import pytorch_lightning as pl
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AdamW, RobertaTokenizer, RobertaForSequenceClassification, AutoModelForSequenceClassification, AutoConfig

from plan_simp.data.roberta import RobertaDataModule
from plan_simp.data.utils import OP_TOKENS, M_OP_TOKENS, READING_LVLS, prepend_tokens
from plan_simp.models.custom_roberta import RobertaForContextualSequenceClassification, ContextRobertaConfig

INF_PARAMS = ["add_context", "context_window", "doc_pos_embeds", "num_labels"]


def load_planner(model_ckpt, add_context=False, tokenizer=None, device="cuda", return_ptl=False):
    """
    Loads and prepares any RoBERTa-based planning model and associated components.
    """

    # prepare AutoClass to handle custom models
    AutoConfig.register("context-roberta", ContextRobertaConfig)
    AutoModelForSequenceClassification.register(ContextRobertaConfig, RobertaForContextualSequenceClassification)

    # load model from file if needed
    if isinstance(model_ckpt, str):
        print("Loading planner model...")
        if model_ckpt.endswith(".ckpt"):
            # load from PytorchLightning trainined checkpoint
            model = RobertaClfFinetuner.load_from_checkpoint(model_ckpt, add_context=add_context).to(device).eval()
        else:
            # load directly from HuggingFace pretrained model
            model = AutoModelForSequenceClassification.from_pretrained(model_ckpt).to(device).eval()
            tokenizer = RobertaTokenizer.from_pretrained(model_ckpt)
    else:
        model = model_ckpt # allow explicit passing of model object

    # unpack model and accompanying components
    hparams = {}
    if isinstance(model, RobertaClfFinetuner):
        # extract what we need from PytorchLightning checkpoint before reducing to HF model
        tokenizer = model.tokenizer
        hparams = model.hparams
        if not return_ptl:
            model.model.config.update(hparams) # directly update internal model config
            model = model.model
    elif isinstance(model, RobertaForContextualSequenceClassification) or isinstance(model, RobertaForSequenceClassification):
        # handle HF models (tokenizer will need to be manually passed as function argument)
        hparams = {p: getattr(model.config, p, None) for p in INF_PARAMS} # extract inference hparams from config file
        if tokenizer is None:
            raise ValueError("A HuggingFace pretrained model must be accompanied by a tokenizer!")

    return model, tokenizer, hparams


def run_classifier(model_ckpt, test_set, x_col="complex", max_samples=None, tokenizer=None, device="cuda", batch_size=16, num_workers=32, 
                    add_context=False, context_dir=None, context_doc_id="pair_id", simple_context_dir=None, simple_context_doc_id="pair_id", 
                    reading_lvl=None, src_lvl=None, simple_sent_id="simp_sent_id", return_logits=False, silent=False):
    if max_samples is not None:
        test_set = test_set[:max_samples]

    # load and prepare planning model
    model, tokenizer, hparams = load_planner(model_ckpt, add_context=add_context, tokenizer=tokenizer, device=device)

    # inference protocol
    with torch.no_grad():
        dm = RobertaDataModule(tokenizer, params=hparams)
        dm.hparams.context_dir = context_dir
        dm.hparams.simple_context_dir = simple_context_dir

        # prepare batch with no labels
        input_seqs = test_set if isinstance(test_set, list) else test_set[x_col].tolist()
        if reading_lvl is not None:
            # prepend reading lvl control tokens
            input_seqs = prepend_tokens(test_set, x_col, lvl_col=reading_lvl)
        if add_context or dm.has_param("add_context"):
            context_ids = dm.prepare_context_ids(test_set, context_doc_id)
            input_seqs = [input_seqs, context_ids]

            # dynamic context
            if simple_context_dir is not None:
                simple_context_ids = dm.prepare_context_ids(test_set, simple_context_doc_id, simple_sent_id)
                input_seqs.append(simple_context_ids)

            # document positional embeddings
            if "doc_pos_embeds" in hparams and hparams["doc_pos_embeds"]:
                doc_positions = list(test_set[["doc_pos", "doc_len"]].itertuples(index=False, name=None))
                input_seqs.append(doc_positions)

            input_seqs = list(zip(*input_seqs))
        else:
            input_seqs = [[x] for x in input_seqs]

        # preprocess data
        test_data = DataLoader(input_seqs, batch_size=batch_size, num_workers=num_workers, collate_fn=dm.prepro_collate)

        # run predictions for each batch
        print("Running predictions...")
        preds = []
        prog = tqdm(test_data) if not silent else test_data
        for batch in prog:
            batch = { k: xi.to(device, non_blocking=True) for k, xi in batch.items() }
            output = model(**batch, return_dict=True)
            logits = output["logits"]
            if return_logits:
                preds += logits
            else:
                preds += [int(l.argmax()) for l in logits]
    
    return preds


class RobertaClfFinetuner(pl.LightningModule):

    def __init__(self, model_name_or_path='roberta-base', tokenizer=None, add_context=False, params=None):
        super().__init__()

        # saves params to the checkpoint and in self.hparams
        self.save_hyperparameters(params)

        num_labels = 5 if self.has_param("multi_split") else 4
        num_labels = 1 if self.has_param("regression") else num_labels
        self.hparams["num_labels"] = num_labels
        self.op_tokens = OP_TOKENS if not self.has_param("multi_split") else M_OP_TOKENS

        if not add_context:
            self.model = RobertaForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
            print(f"Initial RobertaForSequenceClassification model loaded from {model_name_or_path}.")
        else:
            # use contextual model
            doc_pos_embeds = self.has_param("doc_pos_embeds")
            no_context_pos = self.has_param("no_context_pos")
            self.model = RobertaForContextualSequenceClassification.from_pretrained(model_name_or_path, 
                                        num_labels=num_labels, no_context_pos=no_context_pos, doc_pos_embeds=doc_pos_embeds)
            print(f"Initial RobertaForContextualSequenceClassification model loaded from {model_name_or_path}.")

        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
            self.add_new_tokens()
        else:
            self.tokenizer = tokenizer

        # training loss cache to log mean every n steps
        self.train_losses = []

        if "hidden_dropout_prob" in self.hparams and self.hparams.hidden_dropout_prob is not None:
            self.model.config.hidden_dropout_prob = self.hparams.hidden_dropout_prob

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def training_step(self, batch, batch_idx):
        output = self.model(**batch, return_dict=True)
        loss = output["loss"]
        self.train_losses.append(loss)

        # logging mean loss every `n` steps
        if batch_idx % int(self.hparams.train_check_interval * self.trainer.num_training_batches) == 0:
            avg_loss = torch.stack(self.train_losses).mean()
            self.log("train_loss", avg_loss)
            self.train_losses = []

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch, return_dict=True)
        loss = output["loss"]
        logits = output["logits"].cpu()

        output = {
            "loss": loss,
            "preds": logits,
        }

        if not self.has_param("regression"):
            macro_f1 = precision_recall_fscore_support(batch["labels"].cpu(), logits.argmax(axis=1), average="macro")[2]
            output["macro_f1"] = macro_f1

        # accumalte relative acc for each class
        if self.hparams.log_class_acc:
            _, _, class_f1s, sups = precision_recall_fscore_support(batch["labels"].cpu(), logits.argmax(axis=1), average=None, labels=range(self.model.num_labels))

            # if no examples from a class in batch, set f1 to nan so we can ignore in mean calculation
            for i in range(len(sups)):
                if sups[i] == 0:
                    class_f1s[i] = np.nan

            output["accs"] = class_f1s

        return output

    def validation_epoch_end(self, outputs, prefix="val"):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(f"{prefix}_loss", loss)

        if not self.has_param("regression"):
            macro_f1 = np.stack([x["macro_f1"] for x in outputs]).mean()
            self.log(f"{prefix}_macro_f1", macro_f1)
        
        # log relative performance for each class
        if self.hparams.log_class_acc:
            f1s = np.stack([x["accs"] for x in outputs])
            f1s = np.nanmean(f1s, axis=0) # ignore nans in calculation
            for i in range(self.model.num_labels):
                self.log(f"{prefix}_{i}_f1", f1s[i])

        return {f"{prefix}_loss": loss}

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.01
                    },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.0
                    },
                ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
        
        # use a learning rate scheduler if specified
        if self.hparams.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }

        return optimizer

    def add_new_tokens(self):
        self.tokenizer.add_tokens(READING_LVLS + self.op_tokens, special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def has_param(self, param):
        """Check if param exists and has a non-negative/null value."""
        if param in self.hparams:
            param = self.hparams[param] # set `param` to actual value
            if param is not None:
                if not isinstance(param, bool) or param:
                    return True
        return False

    def save_model(self, path):
        # add inference parameters to model config
        self.model.config.update({p: self.hparams[p] for p in INF_PARAMS})

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"{type(self.model)} model saved to {path}.")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument("--name", type=str, default=None, required=False,)
        parser.add_argument("--project", type=str, default=None, required=False,)
        parser.add_argument("--save_dir", type=str, default=None, required=False,)
        parser.add_argument("--checkpoint", type=str, default=None, required=False,)
        parser.add_argument("--wandb_id", type=str, default=None, required=False,)
        
        parser.add_argument("--train_file", type=str, default=None, required=False)
        parser.add_argument("--val_file", type=str, default=None, required=False)
        parser.add_argument("--x_col", type=str, default="x", required=False,)
        parser.add_argument("--y_col", type=str, default="y", required=False,)
        parser.add_argument("--train_check_interval", type=float, default=0.20)
        parser.add_argument("--train_split", type=float, default=0.9)
        parser.add_argument("--val_split", type=float, default=0.05)
        parser.add_argument("--log_class_acc", action="store_true", default=False)
        parser.add_argument("--reading_lvl", type=str, default=None, required=False)
        parser.add_argument("--src_lvl", type=str, default=None, required=False)

        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--max_samples", type=int, default=-1)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--train_workers", type=int, default=8)
        parser.add_argument("--val_workers", type=int, default=8)
        parser.add_argument("--max_length", type=int, default=128)
        parser.add_argument("--lr_scheduler", action="store_true")
        parser.add_argument("--upsample_classes", action="store_true")
        parser.add_argument("--ckpt_metric", type=str, default="val_loss", required=False,)

        parser.add_argument("--hidden_dropout_prob", type=float, default=None, required=False,)

        parser.add_argument("--add_context", action="store_true")
        parser.add_argument("--no_context_pos", action="store_true")
        parser.add_argument("--doc_pos_embeds", action="store_true")
        parser.add_argument("--context_window", type=int, default=5)
        parser.add_argument("--left_z_only", action="store_true")
        parser.add_argument("--context_doc_id", type=str, default=None, required=False,)
        parser.add_argument("--context_dir", type=str, default=None, required=False,)
        parser.add_argument("--simple_context_doc_id", type=str, default=None, required=False,)
        parser.add_argument("--simple_context_dir", type=str, default=None, required=False,)
        parser.add_argument("--binary_clf", action="store_true")

        parser.add_argument("--regression", action="store_true")

        return parser
