#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: utils.py
# Created Date: Thursday, February 4th 2021, 11:31:17 pm
# Author: Chirag Raman
#
# Copyright (c) 2021 Chirag Raman
###


import pickle
from argparse import Namespace, ArgumentParser
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Optional, Sequence, List, Type

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

from common.utils import EnumAction
from data.types import ComponentType, ModelType
from data.loader import BucketSampler
from lightning.builders import RecurrentBuilder, MLPBuilder
from lightning.data import SPSocialDataModule
from lightning.processes import (
    SPSystemSocial, NPSystemBase, VariationalSeq2SeqSystem, CHECKPOINT_MONITOR_METRIC
)


def init_model(
        args: Namespace, ckpt_path: Optional[Path] = None,
        sp_cls: Type = SPSystemSocial
    ) -> LightningModule:
    """ Initialize the Lightning module for the model """
    if args.model == ModelType.SOCIAL_PROCESS:
        # Choose the builder
        if args.component == ComponentType.RNN:
            builder = RecurrentBuilder
        elif args.component == ComponentType.MLP:
            builder = MLPBuilder
        else:
            raise ValueError("Unrecognized component type: expected RNN or MLP")

        # Initialize the SocialProcess model or load from checkpoint
        if ckpt_path is None:
            system = sp_cls(args, builder)
        else:
            system = sp_cls.load_from_checkpoint(str(ckpt_path), builder=builder)
    elif args.model == ModelType.NEURAL_PROCESS:
        # Initialize the NeuralProcess model
        if ckpt_path is None:
            system = NPSystemBase(args)
        else:
            system = NPSystemBase.load_from_checkpoint(str(ckpt_path))
    elif args.model == ModelType.VAE_SEQ2SEQ:
        if ckpt_path is None:
            system = VariationalSeq2SeqSystem(args)
        else:
            system = VariationalSeq2SeqSystem.load_from_checkpoint(str(ckpt_path))
    else:
        raise ValueError("Unsupported model type for training")

    return system


def average_ckpts(ckpts: Sequence, args: Namespace) -> LightningModule:
    """ Return model with weights averaged from different checkpoints """
    # Load the module using the first checkpoint
    model = init_model(args, ckpts[0])

    # Add the params
    params_keys = None
    params_dict = OrderedDict()
    for ckpt in ckpts:
        state = torch.load(ckpt)["state_dict"]
        # Safety check keys
        ckpt_keys = list(state.keys())
        if params_keys is None:
            params_keys = ckpt_keys
        elif params_keys != ckpt_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(ckpt, params_keys, ckpt_keys)
            )
        # Add the params
        for k in params_keys:
            p = state[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # clone() is needed in case p is a shared parameter
            else:
                params_dict[k] += p

    # Average the params
    nckpts = len(ckpts)
    for k in params_dict.keys():
        if params_dict[k].is_floating_point():
            params_dict[k].div_(nckpts)
        else:
            params_dict[k] //= nckpts

    # Load the new state dict
    model.load_state_dict(params_dict)

    return model


def init_data(
        args: Namespace, dataset_dir: Path, load_test_batches: bool = False,
        test_batches_fname: str = "test_batches_random_ctx.pkl"
    ) -> SPSocialDataModule:
    """ Initialize the data module """
    if load_test_batches:
        with open(dataset_dir/test_batches_fname, "rb") as f:
            test_batches = pickle.load(f)
    else:
        test_batches = None
    dm = SPSocialDataModule(dataset_dir, args, test_batches=test_batches)
    return dm


def serialize_test_batches(
        dm: SPSocialDataModule, batch_size: int, dataset_dir: Path, outfile: str
    ) -> None:
    """ Create a sampler and write the test batches to file.

    Filters batches of length less than batch_size

    """
    sampler = BucketSampler(dm.test_set, batch_size)
    batches = sampler.compute_batches()
    print(len(batches), Counter([len(l) for l in batches]))
    filtered = [b for b in batches if len(b) == batch_size]
    print(len(filtered), Counter([len(l) for l in filtered]))
    with open(dataset_dir/outfile, "wb") as f:
        pickle.dump(filtered, f)


def init_ckpt_callbacks(args: Namespace, ckpt_dir: str) -> List:
    """ Initialize checkpoint callbacks """
    callbacks = []

    # Last checkpoint
    last_ckpt_fname = "last-{epoch:03d}"
    last_ckpt = ModelCheckpoint(dirpath=ckpt_dir, filename=last_ckpt_fname)
    callbacks.append(last_ckpt)

    # Monitored checkpoint
    if not args.skip_monitoring:
        mon_ckpt_fname = f"mon-{{epoch}}-{{{CHECKPOINT_MONITOR_METRIC}:.3f}}"
        monitored_ckpt = ModelCheckpoint(
            dirpath=ckpt_dir, filename=mon_ckpt_fname, save_top_k=5,
            mode="min", monitor=CHECKPOINT_MONITOR_METRIC, verbose=True
        )
        callbacks.append(monitored_ckpt)

    return callbacks


def override_hidden_dims(args: Namespace, hidden_dim: int) -> Namespace:
    """ Override the represenation dimensions in the args """
    # Override hidden dimensions
    args.enc_nhid = args.dec_nhid = args.pooler_nhid_embed = args.pooler_nhid_pool \
        = args.pooler_temporal_nhid = hidden_dim
    return args


def add_commmon_args(parent_parser: ArgumentParser) -> ArgumentParser:
    """ Add common args for a driver script """
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("-s", "--seed", type=int, default=1234, help="seed for initializing pytorch")
    parser.add_argument("--dataset_root", type=str, help="root dirctory for dataset")
    parser.add_argument("--model", type=ModelType,
                        action=EnumAction, default=ModelType.SOCIAL_PROCESS,
                        help="type of model to train, default-social process")
    parser.add_argument("--component", type=ComponentType,
                        action=EnumAction, default=ComponentType.RNN,
                        help="type of component modules, default-rnn")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level, default: %(default)s)")
    parser.add_argument("--log_file", type=str, default="logs.txt",
                        help="filename for the log file for metrics")
    return parser