import time
import argparse
from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from sacrebleu import corpus_bleu
from torch.utils.data import DataLoader
from transformers import BartTokenizerFast, AutoModelForSeq2SeqLM, BartForConditionalGeneration, LEDForConditionalGeneration, AutoConfig, AutoTokenizer

from plan_simp.utils import lmap
from plan_simp.data.bart import BartDataModule
from plan_simp.data.utils import CLASS_LABELS, M_CLASS_LABELS, OP_TOKENS, M_OP_TOKENS, READING_LVLS, PLAN_TOKENS, prepend_tokens
from plan_simp.models.custom_bart import BartForContextualGeneration, ContextBartConfig

INF_PARAMS = ["class_labels", "op_tokens", "max_length", "add_context", "context_window", "plan_prefix", "plan_sep"]


def load_simplifier(model_ckpt, tokenizer=None, device="cuda", return_ptl=False):
    """
    Loads and prepares any BART-based simplification model and associated components.
    """

    # prepare AutoClass to handle custom models
    AutoConfig.register("context-bart", ContextBartConfig)
    AutoModelForSeq2SeqLM.register(ContextBartConfig, BartForContextualGeneration)

    if isinstance(model_ckpt, str):
        print("Loading simplification model...")
        if model_ckpt.endswith(".ckpt"):
            # load from PytorchLightning trainined checkpoint
            model = BartFinetuner.load_from_checkpoint(model_ckpt).to(device).eval()
        else:
            # load directly from HuggingFace pretrained model
            model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt, return_dict=True).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    else:
        model = model_ckpt # allow explicit passing of model object

    # unpack model and accompanying components
    hparams = {}
    if isinstance(model, BartFinetuner):
        # extract what we need from PytorchLightning checkpoint before reducing to HF model
        tokenizer = model.tokenizer
        hparams = model.hparams
        if not return_ptl:
            model.model.config.update(hparams) # directly update internal model config
            model = model.model
    elif isinstance(model, BartForConditionalGeneration) or isinstance(model, LEDForConditionalGeneration) or isinstance(BartForContextualGeneration):
        hparams = {p: getattr(model.config, p, None) for p in INF_PARAMS}
        if tokenizer is None:
            raise ValueError("A HuggingFace pretrained model must be accompanied by a tokenizer!")

    return model, tokenizer, hparams


def run_generator(model_ckpt, test_set, x_col="complex", op_col=None, reading_lvl=None,
                    tokenizer=None, max_samples=None, device="cuda", batch_size=16, 
                    num_workers=32, beams=5, length_penalty=1.0, min_length=False, silent=False,
                    context_dir=None, context_doc_id="pair_id", simple_context_dir=None, 
                    simple_context_doc_id="pair_id", simple_sent_id="simp_sent_id"):
    if max_samples is not None:
        test_set = test_set[:max_samples]

    model, tokenizer, hparams = load_simplifier(model_ckpt, tokenizer=tokenizer, device=device)

    with torch.no_grad():
        if op_col is not None or reading_lvl is not None:
            # add control tokens
            input_seqs = prepend_tokens(test_set, x_col, hparams["class_labels"], hparams["op_tokens"], op_col, reading_lvl)
        else:
            input_seqs = test_set if isinstance(test_set, list) else test_set[x_col].tolist()

        # preprocess data
        dm = BartDataModule(tokenizer, params=hparams)
        dm.hparams.context_dir = context_dir
        dm.hparams.simple_context_dir = simple_context_dir

        inputs = [input_seqs]

        # load contextual representations
        if dm.has_param("add_context"):
            context_ids = dm.prepare_context_ids(test_set, context_doc_id)
            inputs.append(context_ids)
            if simple_context_dir is not None:
                simple_context_ids = dm.prepare_context_ids(test_set, simple_context_doc_id, simple_sent_id)
                inputs.append(simple_context_ids)
            pos_info = list(test_set[["doc_pos", "doc_len"]].itertuples(index=False, name=None))
            inputs.append(pos_info)

        inputs.append(["" for _ in input_seqs]) # empty labels
        inputs = list(zip(*inputs))

        test_data = DataLoader(inputs, batch_size=batch_size, num_workers=num_workers, collate_fn=dm.prepro_collate)

        # predict output sequences
        pred_ys = []
        prog = tqdm(test_data) if not silent else test_data
        for i, batch in enumerate(prog):
            batch = {k: xi.to(device, non_blocking=True) for k, xi in batch.items() if k != "labels"}

            _min_length = min_length[i] if isinstance(min_length, list) else min_length
            generated_ids = model.generate(
                **batch,
                use_cache=True,
                decoder_start_token_id=None, # None is default and so will be handled internally
                num_beams=beams,
                max_length=1024,
                min_length=_min_length,
                length_penalty=length_penalty,
            )
            
            # convert generated token ids to text
            skip_specials = not (dm.has_param("plan_prefix") or dm.has_param("plan_sep")) # don't skip when also planning
            gen_text = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=skip_specials,
                clean_up_tokenization_spaces=True)
            results = lmap(str.strip, gen_text)
            
            pred_ys += results

    return pred_ys


class BartFinetuner(pl.LightningModule):

    loss_names = ["loss"]
    metric_names = ["bleu"]

    def __init__(self, model_name_or_path='facebook/bart-base', tokenizer=None, params=None):
        super().__init__()

        # saves params to the checkpoint and in self.hparams
        self.save_hyperparameters(params)
        self.decoder_start_token_id = None

        self.class_labels = CLASS_LABELS if not self.has_param("multi_split") else M_CLASS_LABELS
        self.op_tokens = OP_TOKENS if not self.has_param("multi_split") else M_OP_TOKENS
        self.hparams["class_labels"], self.hparams["op_tokens"] = self.class_labels, self.op_tokens

        # handle Longformer option NOTE: BartTokenizer should still work for Longformer
        if self.has_param("longformer"):
            if model_name_or_path == "facebook/bart-base":
                model_name_or_path = "allenai/led-base-16384"

        if self.has_param("add_context"):
            self.model = BartForContextualGeneration.from_pretrained(model_name_or_path, return_dict=True)
            print(f"Initial {type(self.model)} model loaded from {model_name_or_path}.")
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, return_dict=True)
            print(f"Initial {type(self.model)} model loaded from {model_name_or_path}.")

        if tokenizer is None:
            self.tokenizer = BartTokenizerFast.from_pretrained(model_name_or_path, add_prefix_space=True)
            self.add_new_tokens()
        else:
            self.tokenizer = tokenizer

        # training loss cache to log mean every n steps
        self.train_losses = []

        if self.has_param("plan_loss"):
            self.loss_names = ["loss", "gen_loss", "plan_loss"]
            # get vocab ids for plan operation tokens
            self.plan_op_tok_ids = self.tokenizer.convert_tokens_to_ids(self.op_tokens)

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def _step(self, batch):
        if isinstance(batch, dict):
            labels = batch["labels"]
        
        # run model and get the logits
        outputs = self(**batch, use_cache=False)
        lm_logits = outputs["logits"]

        # compute loss
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad)
        gen_loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))

        # compute planning loss
        if self.has_param("plan_loss"):
            plan_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            plan_masked_lm_loss = plan_loss_fct(lm_logits.view(-1, self.model.config.vocab_size), labels.view(-1))
            plan_token_mask = torch.isin(labels.view(-1), torch.tensor(self.plan_op_tok_ids, device="cuda"))
            plan_loss = plan_masked_lm_loss[plan_token_mask].mean()

            comb_loss = (gen_loss * (1-self.hparams["plan_loss"])) + (plan_loss * self.hparams["plan_loss"])

            return (comb_loss, gen_loss, plan_loss)

        return (gen_loss,)

    def training_step(self, batch, batch_idx) -> Dict:
        result = self._step(batch)
        loss = result[0] # will be either normal loss or combined MTL loss

        self.train_losses.append(loss)

        # logging mean loss every `n` steps
        if batch_idx % int(self.hparams.train_check_interval * self.trainer.num_training_batches) == 0:
            avg_loss = torch.stack(self.train_losses).mean()
            self.log("train_loss", avg_loss)
            self.train_losses = []

        return {"loss": loss}
        
    def validation_step(self, batch, batch_idx) -> Dict:
        if self.hparams.skip_val_gen:
            loss_tensors = self._step(batch)
            val_results = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        else:
            val_results = self._generative_step(batch)

        return val_results

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        # compile val loss/metrics and calculate aggregates
        losses = {k: torch.stack([x[k] for x in outputs]).mean()
                  for k in self.loss_names}
        loss = losses["loss"]
        result = {f"{prefix}_{loss_name}": v for loss_name, v in losses.items()}

        if not self.hparams.skip_val_gen:
            # add generative metric summaries to losses
            generative_metrics = {k: np.array([x[k] for x in outputs]).mean() 
                                    for k in self.metric_names + ["gen_time", "gen_len"]}
            metric_val = (generative_metrics[self.hparams.val_metric]
                        if self.hparams.val_metric in generative_metrics else losses[self.hparams.val_metric])
            metric_tensor: torch.FloatTensor = torch.tensor(metric_val).type_as(loss)
            result[f"{prefix}_{self.hparams.val_metric}"] = metric_tensor

            generative_metrics.update({k: v.item() for k, v in losses.items()})
            losses.update(generative_metrics)
        
        # wandb log
        for loss_name, v in result.items():
            self.log(loss_name, v)

        return result

    def _generative_step(self, batch):
        t0 = time.time()
        
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # generate sequences from batch input
        generated_ids = self.model.generate(
            **{k: xi for k, xi in batch.items() if k != "labels"},
            use_cache=True,
            decoder_start_token_id=self.decoder_start_token_id,
            num_beams=self.hparams.eval_beams,
            max_length=self.hparams.eval_max_length,
        )
        gen_time = (time.time() - t0) / input_ids.shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(labels)

        # compute loss
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}

        # calculate other metrics
        bleu: Dict = self.calc_generative_metrics(preds, target)
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(
            gen_time=gen_time,
            gen_len=summ_len,
            preds=preds,
            target=target,
            **bleu)

        return base_metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

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
        if self.has_param("plan_prefix") and not self.has_param("sent_level"):
             self.tokenizer.add_tokens(PLAN_TOKENS)
        self.model.resize_token_embeddings(len(self.tokenizer))

    # UTILITY FUNCTIONS #

    def calc_generative_metrics(self, preds, target) -> dict:
        bleu = round(corpus_bleu(preds, [target]).score, 4)
        return {"bleu": bleu}

    def ids_to_clean_text(self, generated_ids: List[int]):
        """Decodes generated token ids intro Strings and strips result."""
        gen_text = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)
        return lmap(str.strip, gen_text)

    def shift_tokens_right(self, input_ids, pad_token_id):
        """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
        prev_output_tokens = input_ids.clone()
        index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
        prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
        prev_output_tokens[:, 1:] = input_ids[:, :-1]
        return prev_output_tokens

    def save_tokenizer(self, path):
        self.tokenizer.save_pretrained(path)
        print(f"Tokenizer saved to {path}.")

    def save_model(self, path):
        # add inference parameters to model config
        self.model.config.update({p: self.hparams.get(p) for p in INF_PARAMS})

        self.model.save_pretrained(path)
        self.save_tokenizer(path)
        print(f"{type(self.model)} model saved to {path}.")

    def load_tokenizer(self, path):
        self.tokenizer = BartTokenizerFast.from_pretrained(path, add_prefix_space=True)
        print(f"Tokenizer loaded from {path}.")

    def load_model(self, path):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path, return_dict=True)
        print(f"{type(self.model)} model loaded from {path}.")

    def has_param(self, param):
        """Check if param exists and has a non-negative/null value."""
        if param in self.hparams:
            param = self.hparams[param] # set `param` to actual value
            if param is not None:
                if not isinstance(param, bool) or param:
                    return True
        return False

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
        parser.add_argument("--train_check_interval", type=float, default=0.01)
        parser.add_argument("--train_split", type=float, default=0.9)
        parser.add_argument("--val_split", type=float, default=0.05)
        parser.add_argument("--train_data_dir", type=str, default=None, required=False,)
        parser.add_argument("--valid_data_dir", type=str, default=None, required=False,)

        parser.add_argument("--max_samples", type=int, default=-1)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--lr", type=float, default=2e-5)
        parser.add_argument("--lr_scheduler", action="store_true")
        parser.add_argument("--train_workers", type=int, default=8)
        parser.add_argument("--val_workers", type=int, default=8)

        parser.add_argument("--max_length", type=int, default=1024)
        parser.add_argument("--skip_val_gen", action="store_true", default=False)
        parser.add_argument("--eval_beams", type=int, default=None, required=False)
        parser.add_argument("--val_metric", type=str, default="bleu", required=False,
            choices=["bleu", "rouge2", "loss",None])
        parser.add_argument("--eval_max_length", type=int, default=None)

        parser.add_argument("--reading_lvl", type=str, default=None, required=False)
        parser.add_argument("--op_col", type=str, default=None, required=False)
        parser.add_argument("--sent_level", action="store_true")

        parser.add_argument("--longformer", action="store_true")

        parser.add_argument("--add_context", action="store_true")
        parser.add_argument("--context_window", type=int, default=13)
        parser.add_argument("--context_doc_id", type=str, default=None, required=False,)
        parser.add_argument("--context_dir", type=str, default=None, required=False,)
        parser.add_argument("--simple_context_doc_id", type=str, default=None, required=False,)
        parser.add_argument("--simple_context_dir", type=str, default=None, required=False,)

        parser.add_argument("--plan_prefix", action="store_true")
        parser.add_argument("--prefix_only", action="store_true")
        parser.add_argument("--plan_loss", type=float, default=None)
        parser.add_argument("--plan_sep", action="store_true")
        parser.add_argument("--plan_col", type=str, default="labels", required=False,)

        return parser
