#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: test_.py
# Created Date: Tuesday, January 5th 2021, 1:00:07 pm
# Author: Chirag Raman
#
# Copyright (c) 2021 Chirag Raman
###


import argparse
import logging
import pickle
from argparse import Namespace
from enum import Enum, auto
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TestTubeLogger

from common.initialization import init_torch
from common.utils import EnumAction, configure_logging
from common.model import summarize
from data.datasets import SocialDataset
from data.types import (
    ModelType, ComponentType, ContextRegime, SerializationMap
)
from lightning.callbacks import MetricsComputer, TestSerializer
from lightning.data import SPSocialDataModule, COLUMNS_TO_STDIZE
from run.utils import average_ckpts, init_model, init_data


class ModelLoadProc(Enum):

    """ Enum for denoting type of model to train """

    CKPT  = auto()
    AVG_CKPTS  = auto()


def parse_serializiation_map(arg: str) -> SerializationMap:
    """ Convert string to map of group_id to list of tuples of obs starts

    Value for a group_id can be -1 for serializing all sequences or a sequence
    of even number of comma separated entries representing (start,end) tuples
    of ranges of observation start frames. Multiple groups are expected to be
    semicolon separated.
    Eg. of input string "<g_id1>:-1;<g_id2>:<fr1>,<fr2>,<fr3>,<fr4>"

    """
    groups = [g.split(":") for g in arg.split(";")]
    s_map = dict()
    for g in groups:
        if len(g) != 2:
            raise ValueError("Invalid entry for a group, "
                             "expected to be : separated")
        key = g[0]
        values = [int(v) for v in g[1].split(",")]
        if ((len(values) == 1 and values[0] != -1)
                or (len(values) != 1 and len(values) % 2 != 0)):
            raise ValueError("Expected even number of frame entries per group,"
                             " or -1.")
        if len(values) == 1:
            values = values[0]
        else:
            it = iter(values)
            values = list(zip(it, it))
        s_map[key] = values
    return s_map


def main():
    """ Run the main experiment """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-s", "--seed", type=int, default=1234,
                        help="seed for initializing pytorch")
    parser.add_argument("--load_proc", type=ModelLoadProc,
                        action=EnumAction, default=ModelLoadProc.CKPT,
                        help="procedure for loading model; ckpt, averaging ckpts, "
                              "or loading state dict. Default CKPT")
    parser.add_argument("--ckpt_path", type=str,
                        help="path to the model checkpoint; used if `load_proc` "
                             "is CKPT.")
    parser.add_argument("--ckpt_dir", type=str,
                        help="directory that contains checkpoints to average;"
                             "only used if `load_proc` is AVG_CKPTS")
    parser.add_argument("--model", type=ModelType,
                        action=EnumAction, default=ModelType.SOCIAL_PROCESS,
                        help="type of model to train, default-social process")
    parser.add_argument("--component", type=ComponentType,
                        action=EnumAction, default=ComponentType.RNN,
                        help="type of component modules, default-rnn")
    parser.add_argument("--results_dir", type=str, default="test_results",
                        help="root output dir name")
    parser.add_argument("--serialize_batch", default=False, action="store_true",
                        help="serialize the batch and predictions")
    parser.add_argument("--skip_serialize_seq", default=False,
                        action="store_true",
                        help="skip serialization of individual sequences")
    parser.add_argument("--serialization_map", default=None,
                        type=parse_serializiation_map,
                        help="map from group_id to list of tuples denoting "
                             "start frames of observed sequences to serialize")
    parser.add_argument("--project_rot", default=False, action="store_true",
                        help="compute rotation error by reprojecting tracking "
                             "direction to xy plane")
    parser.add_argument("--skip_serialize_cb", default=False, action="store_true",
                        help="serialize the sequences serialization callback")
    parser.add_argument("--skip_metrics_cb", default=False, action="store_true",
                        help="serialize the metrics computation callback")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level, default: %(default)s)")
    parser.add_argument("--log_file", type=str, default="test_logs.txt",
                        help="filename for the log file for metrics")

    parser = Trainer.add_argparse_args(parser)
    parser = SPSocialDataModule.add_data_args(parser)
    parser = SocialDataset.add_dataset_specific_args(parser)
    args = parser.parse_args()

    # Initialize pytorch
    init_torch(args.seed)
    seed_everything(args.seed)

    # Set path vars and create the output directory and log file
    artefacts_dir = (Path(__file__).resolve().parent.parent / "artefacts/")

    if args.load_proc == ModelLoadProc.CKPT:
        ckpt_path = Path(args.ckpt_path)
        outroot = ckpt_path.parents[2]
    elif args.load_proc == ModelLoadProc.AVG_CKPTS:
        ckpt_dir = Path(args.ckpt_dir)
        outroot = ckpt_dir.parents[2]

    out_dir = outroot / args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    args.log_file = str(out_dir / args.log_file)

    # Setup logging
    configure_logging(args.log_level, args.log_file)

    # Load the model(s)
    if args.load_proc == ModelLoadProc.CKPT:
        process = init_model(args, ckpt_path)
    elif args.load_proc == ModelLoadProc.AVG_CKPTS:
        logging.info("[*] Averaging checkpoints")
        ckpts = list(ckpt_dir.glob("**/*"))
        if len(ckpts) == 0:
            raise ValueError("Expecting checkpoints to average, found none")
        process = average_ckpts(ckpts, args)
        torch.save(process.state_dict(),
                   ckpt_dir.parents[0] / "averaged.pt")
    else:
        raise ValueError("Unsupported model loading procedure.")

    # Summarize the module and the parameters
    process.freeze()
    summarize(process)

    # Prepare the data module
    if args.context_regime == ContextRegime.RANDOM:
        batches_fname = "test_batches_random_ctx.pkl"
    else:
        batches_fname = "test_batches_fixed_ctx.pkl"
    dm = init_data(args, load_test_batches=True,
                   test_batches_fname=batches_fname)
    dm.setup("test")

    # Initialize the callbacks:
    # Load train statistics to denormalize
    train_stats = pd.read_hdf(
        artefacts_dir/"datasets/panoptic-haggling/train_description.h5"
    )
    train_mean = (train_stats.loc["mean", COLUMNS_TO_STDIZE]
                  .values.astype(np.float32))
    # Hardcoded for 2 poses, TODO: generalize
    train_mean = torch.from_numpy(np.hstack(
        [[0, 0, 0, 0], train_mean[:3], [0, 0, 0, 0], train_mean[3:], 0]
    ))
    train_std = (train_stats.loc["std", COLUMNS_TO_STDIZE]
                 .values.astype(np.float32))
    # Hardcoded for 2 poses, TODO: generalize
    train_std = torch.from_numpy(np.hstack(
        [[1, 1, 1, 1], train_std[:3], [1, 1, 1, 1], train_std[3:], 1]
    ))
    # Initialize the callbacks
    callbacks = []
    if args.skip_serialize_cb:
        logging.info("[*] Skipping serialization callback")
    else:
        logging.info("[*] Initializaing serialization callback")
        logging.info(f"\t[-] Serialization map: {args.serialization_map}")
        serializer = TestSerializer(
            out_dir/"serialized", args.time_stride, train_mean, train_std,
            args.serialize_batch, not args.skip_serialize_seq,
            args.serialization_map
        )
        callbacks.append(serializer)

    if args.skip_metrics_cb:
        logging.info("[*] Skipping metrics computation callback")
    else:
        logging.info("[*] Initializaing metrics computation callback")
        metrics_computer = MetricsComputer(
            out_dir, process.hparams.nposes, args.future_len, args.time_stride,
            train_mean, train_std, args.project_rot
        )
        callbacks.append(metrics_computer)

    # Create  trainer and test
    trainer = Trainer.from_argparse_args(
        args, logger=False, checkpoint_callback=False,
        callbacks=callbacks
    )

    trainer.test(process, datamodule=dm)


if __name__ == "__main__":
    main()