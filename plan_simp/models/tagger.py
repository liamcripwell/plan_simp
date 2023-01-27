import argparse

import numpy as np
import torch
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from transformers import AdamW, RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForCausalLM, BartForConditionalGeneration, BartTokenizerFast

from plan_simp.data.tagging import TaggerDataModule
from plan_simp.data.utils import OP_TOKENS, M_OP_TOKENS, READING_LVLS
from plan_simp.models.custom_roberta import RobertaForDocumentTokenClassification, EncDecForDocumentTagging


def run_ar_generation(model, tokenizer, batch, input_lens=None, bart=False, op_tokens=OP_TOKENS):
    # prepare non-sentence token embeddings
    if not bart and "ctrl_tok_ids" in batch:
        new_masks = []
        new_embeds = []
        ctrl_tok_embeds = model.encoder.embeddings.word_embeddings(batch["ctrl_tok_ids"])
        for i, x in enumerate(batch["inputs_embeds"]):
            new_embeds.append(torch.cat([ctrl_tok_embeds[i], x]))
            new_masks.append(torch.cat([torch.ones(len(ctrl_tok_embeds[i]), device=batch["attention_mask"].device), batch["attention_mask"][i]]))
        batch["inputs_embeds"] = torch.stack(new_embeds)
        batch["attention_mask"] = torch.stack(new_masks)

    if not bart:
        # generate sequences and convert to predicted labels
        generated = model.generate(
            inputs_embeds=batch["inputs_embeds"], 
            attention_mask=batch["attention_mask"],
            max_length=512,)
    else:
        # BART generation
        generated = model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            num_beams=5,
            max_length=1024,
        )
        
    # decode predicted tokens and reformat into valid \hat{l}
    generated = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    preds = [[op_tokens.index(y_) if y_ in op_tokens else 0 for y_ in ys_.split()] for ys_ in generated]

    if input_lens is None:
        if bart:
            input_lens = batch["num_sents"]
        else:
            # infer lens via size of input - special tokens (ctrl toks and pads)
            # NOTE: this assumes ctrl_tok_ids is not empty
            ctrl_tok_count = 0 if "ctrl_tok_ids" not in batch else batch["ctrl_tok_ids"].shape[1]
            input_lens = [torch.count_nonzero(x.sum(dim=1))-ctrl_tok_count for x in batch["inputs_embeds"]]
    
        preds_trim = [preds[i][:input_lens[i]] for i in range(len(preds))]

    # prepend deletion placeholders if predicted shorter sequence
    for i in range(len(preds)):
        if len(preds_trim[i]) < input_lens[i]:
            preds_trim[i] += [model.num_labels-1]*(input_lens[i]-len(preds_trim[i]))

    return preds_trim, batch


def run_tagger(model_ckpt, test_set, x_col="pair_id", max_samples=None, device="cuda", batch_size=16, num_workers=8,
                bart=False, embed_dir=None, reading_lvl=None, multi_split=False, op_tokens=OP_TOKENS, return_logits=False):
    if max_samples is not None:
        test_set = test_set[:max_samples]

    print("Loading model...")
    model = RobertaTagFinetuner.load_from_checkpoint(model_ckpt).to(device).eval()

    with torch.no_grad():
        # load DataModule and update data directory
        dm = TaggerDataModule(tokenizer=model.tokenizer, params=model.hparams)
        dm.hparams.embed_dir = embed_dir

        # preprocess data
        input_data = [x for x in test_set[x_col].tolist()]
        if reading_lvl is not None:
            # include reading lvl control tokens
            lvls = [READING_LVLS[t] for t in test_set[reading_lvl]]
            input_data = list(zip(input_data, lvls))
        else:
            input_data = [[x] for x in input_data]
        test_data = DataLoader(input_data, batch_size=batch_size, num_workers=num_workers, collate_fn=dm.prepro_collate)

        # run predictions for each batch
        print("Running predictions...")
        preds = []
        for batch in tqdm(test_data):
            batch = { k: xi.to(device, non_blocking=True) for k, xi in batch.items() }

            if not model.has_param("use_decoder"):
                output = model.model(**batch, return_dict=True)
                logits = output["logits"]
                if return_logits:
                    preds += logits
                else:
                    preds += logits.argmax(-1)[:,len(batch["ctrl_tok_ids"][0]):] # assumes constant no. of ctrl tokens
            else:
                generated, batch = run_ar_generation(model.model, model.tokenizer, batch, bart=bart, op_tokens=op_tokens)
                preds += generated

    return preds


class RobertaTagFinetuner(pl.LightningModule):

    def __init__(self, model_name_or_path='roberta-base', params=None):
        super().__init__()

        # saves params to the checkpoint and in self.hparams
        self.save_hyperparameters(params)

        num_labels = 5 if self.has_param("multi_split") else 4
        self.op_tokens = OP_TOKENS if not self.has_param("multi_split") else M_OP_TOKENS

        self.tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)

        self.arg_blacklist = [] # used to exclude batch members from forward pass

        if self.has_param("use_bart"):
            # BART full input option
            model_name_or_path = "facebook/bart-base"
            self.model = BartForConditionalGeneration.from_pretrained(model_name_or_path, return_dict=True)
            self.model.num_labels = num_labels
            print(f"Initial BartForConditionalGeneration model loaded from {model_name_or_path}.")

            self.tokenizer = BartTokenizerFast.from_pretrained(model_name_or_path, add_prefix_space=True)

            self.arg_blacklist = ["num_sents", "ctrl_tok_ids"]
        elif self.has_param("use_decoder"):
            # Auto-regressive option
            if self.has_param("enc_checkpoint"):
                encoder = RobertaModel.from_pretrained(self.hparams.enc_checkpoint)
            else:
                encoder = RobertaModel.from_pretrained("roberta-base")

            dec_config = RobertaConfig.from_pretrained("roberta-base")
            dec_config.is_decoder = True
            dec_config.add_cross_attention = True
            dec_config.num_hidden_layers = self.hparams.decoder_layers
            decoder = RobertaForCausalLM(config=dec_config)

            self.model = EncDecForDocumentTagging(encoder=encoder, decoder=decoder, num_labels=num_labels)
            self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            print(f"Initialized autoregressive tagger with {self.hparams.decoder_layers} layer decoder.")
        else:
            # self.model = RobertaForTokenClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
            # print(f"Initial RobertaForTokenClassification model loaded from {model_name_or_path}.")
            self.model = RobertaForDocumentTokenClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
            print(f"Initial RobertaForDocumentTokenClassification model loaded from {model_name_or_path}.")

        # add special tokens to vocabulary
        self.add_new_tokens()

        # training loss cache to log mean every n steps
        self.train_losses = []

    def forward(self, inputs_embeds, **kwargs):
        # we expect to always already have embeddings before forward pass
        return self.model(inputs_embeds, **kwargs)

    def training_step(self, batch, batch_idx):
        reduced_batch = {k: v for k, v in batch.items() if k not in self.arg_blacklist}

        output = self.model(**reduced_batch, return_dict=True)
        loss = output["loss"]
        self.train_losses.append(loss)

        # logging mean loss every `n` steps
        if batch_idx % int(self.hparams.train_check_interval * self.trainer.num_training_batches) == 0:
            avg_loss = torch.tensor(self.train_losses).mean()
            self.log("train_loss", avg_loss)
            self.train_losses = []

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        reduced_batch = {k: v for k, v in batch.items() if k not in self.arg_blacklist}
        
        output = self.model(**reduced_batch, return_dict=True)
        loss = output["loss"]
        
        # only consider loss if in seq2seq mode
        if self.has_param("use_decoder"):
            with torch.no_grad():
                # seq2seq mode
                preds, batch = run_ar_generation(self.model, self.tokenizer, batch, bart=self.has_param("use_bart"), op_tokens=self.op_tokens)
                preds = [l for x in preds for l in x] # flatten

            output = {"loss": loss}

        if not self.has_param("use_decoder"):
            logits = output["logits"].cpu()

            # flatten preds to evaluate on sentence-level
            preds = logits.argmax(-1)[:,len(batch["ctrl_tok_ids"][0]):].reshape(-1) # assumes constant no. of ctrl tokens
            output["preds"] = logits

        labs = batch["labels"].cpu().reshape(-1) # flatten labels
        if self.has_param("use_decoder"):
            labs = [l for l in labs if l not in self.tokenizer.all_special_ids] # remove special tokens (CLS, pad, etc.)
            labs = [l for l in labs if l not in self.tokenizer(" ".join(READING_LVLS)).input_ids] # remove reading lvls
            labs = self.tokenizer.convert_ids_to_tokens(labs)
            labs = [self.op_tokens.index(l) for l in labs] # convert to correct index values

        macro_f1 = precision_recall_fscore_support(labs, preds, average="macro", labels=range(self.model.num_labels))[2]

        output["macro_f1"] =  macro_f1

        # accumalte relative acc for each class
        if self.hparams.log_class_acc:
            _, _, class_f1s, sups = precision_recall_fscore_support(labs, preds, average=None, labels=range(self.model.num_labels))

            # if no examples from a class in batch, set f1 to nan so we can ignore in mean calculation
            for i in range(len(sups)):
                if sups[i] == 0:
                    class_f1s[i] = np.nan

            output["accs"] = class_f1s

        return output

    def validation_epoch_end(self, outputs, prefix="val"):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(f"{prefix}_loss", loss)

        macro_f1 = np.stack([x["macro_f1"] for x in outputs]).mean()
        self.log(f"{prefix}_macro_f1", macro_f1)
        
        # log relative performance for each class
        if self.has_param("log_class_acc"):
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
        self.tokenizer.add_tokens(READING_LVLS)
        self.tokenizer.add_tokens(self.op_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

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
        parser.add_argument("--train_check_interval", type=float, default=0.20)
        parser.add_argument("--train_split", type=float, default=0.9)
        parser.add_argument("--val_split", type=float, default=0.05)
        parser.add_argument("--log_class_acc", action="store_true", default=False)
        parser.add_argument("--reading_lvl", type=str, default=None, required=False)

        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--max_samples", type=int, default=-1)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--train_workers", type=int, default=8)
        parser.add_argument("--val_workers", type=int, default=8)
        parser.add_argument("--lr_scheduler", action="store_true")
        parser.add_argument("--ckpt_metric", type=str, default="val_loss", required=False,)

        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--hidden_dropout_prob", type=float, default=None, required=False,)

        parser.add_argument("--embed_dir", type=str, default=None, required=False,)
        parser.add_argument("--multi_split", action="store_true")

        parser.add_argument("--use_bart", action="store_true")
        parser.add_argument("--use_decoder", action="store_true")
        parser.add_argument("--decoder_layers", type=int, default=1)
        parser.add_argument("--enc_checkpoint", type=str, default=None, required=False,)

        return parser
