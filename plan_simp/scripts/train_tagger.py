import argparse

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from plan_simp.data.tagging import TaggerDataModule
from plan_simp.models.tagger import RobertaTagFinetuner

if __name__ == '__main__':

    # prepare argument parser
    parser = argparse.ArgumentParser()

    # add model specific args to parser
    parser = RobertaTagFinetuner.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # prepare data module and finetuner class
    if args.checkpoint is None:
        model = RobertaTagFinetuner(params=args)
    else:
        model = RobertaTagFinetuner.load_from_checkpoint(args.checkpoint, params=args)

    dm = TaggerDataModule(tokenizer=model.tokenizer, params=args)

    if args.name is None:
        # use default logger settings (for hparam sweeps)
        wandb_logger = WandbLogger()
        checkpoint_callback=None
    else:
        # prepare logger
        wandb_logger = WandbLogger(
            name=args.name, project=args.project, save_dir=args.save_dir, id=args.wandb_id)

        # checkpoint callback
        mode = "max" if args.ckpt_metric.split("_")[-1] == "f1" else "min"
        checkpoint_callback = [ModelCheckpoint(monitor=args.ckpt_metric, mode=mode, save_last=True)]

    trainer = pl.Trainer.from_argparse_args(
        args,
        val_check_interval=args.val_check_interval,
        logger=wandb_logger,
        accelerator="ddp",
        plugins=DDPPlugin(find_unused_parameters=True), # needs to be set to true because we don't use the roberta embedding layers
        callbacks=checkpoint_callback,
        precision=16,)

    trainer.fit(model, dm)
