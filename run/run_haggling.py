#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: run_synthetic_social.py
# Created Date: Sunday, July 5th 2020, 1:28:08 pm
# Author: Chirag Raman
#
# Copyright (c) 2020 Chirag Raman
###


import argparse
from argparse import Namespace
from pathlib import Path
from typing import List

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TestTubeLogger

import constants.paths as paths
from common.initialization import init_torch
from common.model import summarize
from common.utils import EnumAction, configure_logging
from data.datasets import SocialDataset
from data.types import ModelType, ComponentType
from lightning.data import SPSocialDataModule
from lightning.processes import SPSystemSocial
from run.utils import (
    init_model, init_data, init_ckpt_callbacks, override_hidden_dims
)

def main() -> None:
    """ Run the main experiment """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", type=ModelType,
                        action=EnumAction, default=ModelType.SOCIAL_PROCESS,
                        help="type of model to train, default-social process")
    parser.add_argument("--component", type=ComponentType,
                        action=EnumAction, default=ComponentType.RNN,
                        help="type of component modules, default-rnn")
    parser.add_argument("--skip_monitoring", default=False, action="store_true",
                        help="Skip the monitored checkpoint callback")
    parser.add_argument("--override_hid_dims", default=False, action="store_true",
                        help="Override representation dimensions to `hid_dim`")
    parser.add_argument("--hid_dim", type=int, default=1024,
                        help="dimension to override representations")
    parser.add_argument("-s", "--seed", type=int, default=1234,
                        help="seed for initializing pytorch")
    parser.add_argument("--out_dir", type=str, help="root output directory")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level, default: %(default)s)")
    parser.add_argument("--log_file", type=str, default="logs.txt",
                        help="filename for the log file for metrics")

    # Add Trainer args
    parser = Trainer.add_argparse_args(parser)
    # Add model specific args
    parser = SPSystemSocial.add_model_specific_args(parser)
    parser = SPSocialDataModule.add_data_args(parser)
    parser = SocialDataset.add_dataset_specific_args(parser)
    args = parser.parse_args()

    # Override representations dim if needed
    if args.override_hid_dims:
        args = override_hidden_dims(args, args.hid_dim)

    # Create the output root and update log_file path
    outroot = Path(args.out_dir)
    outroot.mkdir(parents=True, exist_ok=True)
    args.log_file = str(outroot / args.log_file)

    # Setup logging
    configure_logging(args.log_level, args.log_file)

    # Initialize pytorch
    init_torch(args.seed)
    seed_everything(args.seed)

    # Prepare the datasets
    dm = init_data(args)
    dm.setup("fit")

    # Initialize the lightning module
    process = init_model(args)

    # Summarize the module and the parameters
    summarize(process)

    # Initialize checkpoint callbacks and resume ckpt
    ckpt_dir = outroot / paths.LOG_SUBDIR / "checkpoints"
    callbacks = init_ckpt_callbacks(args, str(ckpt_dir))
    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt is not None:
        resume_ckpt = str(ckpt_dir / resume_ckpt)

    # Create experiment
    logger = TestTubeLogger(save_dir=str(outroot / paths.LOG_SUBDIR))

    # Create trainer and fit
    trainer = Trainer.from_argparse_args(
        args, logger=logger, callbacks=callbacks,
        resume_from_checkpoint=resume_ckpt
    )
    trainer.fit(process, datamodule=dm)


if __name__ == "__main__":
    main()
