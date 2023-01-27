import argparse

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from plan_simp.data.bart import BartDataModule
from plan_simp.models.bart import BartFinetuner

if __name__ == '__main__':

    # prepare argument parser
    parser = argparse.ArgumentParser()

    # add model specific args to parser
    parser = BartFinetuner.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # prepare data module and finetuner class
    if args.checkpoint is None:
        model = BartFinetuner(params=args)
    else:
        model = BartFinetuner.load_from_checkpoint(args.checkpoint, params=args, strict=False)

    dm = BartDataModule(model.tokenizer, params=args)

    # construct default run name
    if args.name is None:
        args.name = f"{args.max_samples}_{args.batch_size}_{args.lr}"

    # prepare logger
    wandb_logger = WandbLogger(
        name=args.name, project=args.project, save_dir=args.save_dir, id=args.wandb_id)

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_last=True)

    trainer = pl.Trainer.from_argparse_args(
        args,
        val_check_interval=args.val_check_interval,
        logger=wandb_logger,
        accelerator="ddp",
        plugins=DDPPlugin(find_unused_parameters=True),
        callbacks=[checkpoint_callback],
        precision=16,)

    trainer.fit(model, dm)
