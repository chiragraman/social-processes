#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: run_synthetic_glancing.py
# Created Date: Tuesday, December 17th 2019, 8:13:42 pm
# Author: Chirag Raman
#
# Copyright (c) 2019 Chirag Raman
###


import argparse
import logging
import pickle
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch import Tensor
from torch.distributions import Normal
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TestTubeLogger
from torch.utils.data import DataLoader

import constants.paths as paths
from common.initialization import init_torch
from common.utils import EnumAction, configure_logging
from data.datasets import SyntheticGlancing
from data.loader import collate_sampled_context, collate_seq2seq
from data.types import ModelType, ComponentType, DataSplit, Seq2SeqSamples
from lightning.processes import SPSystemBase
from run.utils import init_model, init_ckpt_callbacks, override_hidden_dims


def train(train_set: SyntheticGlancing, outroot: Path, args: Namespace):
    """ Train the models """
    # Initialize the model
    process = init_model(args, sp_cls=SPSystemBase)

    # Initialize checkpoint callbacks and resume ckpt
    ckpt_dir = outroot / paths.LOG_SUBDIR / "checkpoints"
    callbacks = init_ckpt_callbacks(args, str(ckpt_dir))
    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt is not None:
        resume_ckpt = str(ckpt_dir / resume_ckpt)

    logger = TestTubeLogger(save_dir=str(outroot / paths.LOG_SUBDIR))

    # Create dataloader and pass to trainer
    ncontext = args.batch_size // 4
    loader = DataLoader(
        train_set, shuffle=True, batch_size=args.batch_size,
        collate_fn=lambda x: collate_sampled_context(x, ncontext=ncontext),
    )

    # Create experiment and fit
    trainer = Trainer.from_argparse_args(
        args, logger=logger, callbacks=callbacks,
        resume_from_checkpoint=resume_ckpt
    )
    trainer.fit(process, train_dataloader=loader, val_dataloaders=[loader])


def convert_sine_error_to_angle(
        error: Tensor, lower: float =  0, upper: float = 180
    ) -> Tensor:
    """ Converts an error in the mean prediction to an angle

    Assumes that the [-1, 1] sine amplitude range lies between [lower, upper]

    """
    return error * (upper - lower) / 2


def compute_test_metrics(
        test_set: SyntheticGlancing, ckpt_path: Path, ctx_idxs: np.ndarray,
        resdir: Path, args: Namespace
    ):
    """ Compute the NLL and orientation error for the model """
    # Get predictions from model
    process = init_model(args, ckpt_path, sp_cls=SPSystemBase)
    process.freeze()

    # Include both samples for phase in the context
    # ctx_idx_rep = ctx_idxs
    ctx_idx_rep = np.repeat(ctx_idxs, 2)
    ctx_idx_rep[1::2] += 1

    # Construct context
    context = [test_set.__getitem__(i) for i in ctx_idx_rep]
    context = collate_seq2seq(context)

    # Get all predictions
    target_idxs = list(range(len(test_set)))
    target = [test_set.__getitem__(i) for i in target_idxs]
    target = collate_seq2seq(target)
    split = DataSplit(context=context, target=target)
    preds = process(split)

    # Extract the data tensors
    # target tensors are of of shape [seq_len, batch_size, 1, 1]
    # The predicted future distribution has loc and scale of shape
    # [1, seq_len, batch_size, 1, 1]
    expected = target.future.detach().squeeze(dim=2).squeeze(dim=2).cpu()
    p_dist = preds.stochastic
    future_mean = p_dist.loc.detach()[0].squeeze(dim=2).squeeze(dim=2).cpu()
    future_std = p_dist.scale.detach()[0].squeeze(dim=2).squeeze(dim=2).cpu()

    # Compute the NLL and MSE, and orientation error
    nll = - Normal(future_mean, future_std).log_prob(expected)
    # Take mean over timesteps
    nll_mean = nll.mean(dim=0)

    # Compute the orientation error wrt seq gt
    errors = abs(expected - future_mean)
    error_angles = convert_sine_error_to_angle(errors)
    # Print and save metrics
    angle_means = error_angles.mean(dim=0)
    metric_str = (f"NLL : {nll_mean.mean()}({nll_mean.std()}), "
                  f"angle_error: {angle_means.mean()}({angle_means.std()})")
    print(metric_str)
    # Write to file for pretty plotting
    with open(resdir/"metrics.txt", "w") as f:
        print(metric_str, file=f)


def generate_test_ctx_idxs(phases: np.ndarray, ntest_ctx: int):
    """ Generate context indices for testing """
    rng = np.random.default_rng()
    ctx_idxs = rng.choice(np.arange(len(phases)), ntest_ctx, replace=False)
    # Dataset contains two entries for each phase, double idxs
    ctx_idxs *= 2
    return ctx_idxs


def main() -> None:
    """ Run the main experiment """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-s", "--seed", type=int, default=1234,
                        help="seed for initializing pytorch")
    parser.add_argument("--skip_monitoring", default=False, action="store_true",
                        help="Skip the monitored checkpoint callback")
    parser.add_argument("--model", type=ModelType,
                        action=EnumAction, default=ModelType.SOCIAL_PROCESS,
                        help="type of model to train, default-social process")
    parser.add_argument("--component", type=ComponentType,
                        action=EnumAction, default=ComponentType.RNN,
                        help="type of component modules, default-rnn")
    parser.add_argument("--future_len", type=int, default=20,
                        help="number of future timesteps to predict")
    parser.add_argument("--outdir", type=str,
                        help="root output directory relative to 'artefacts/exp'")
    parser.add_argument("--resdir", type=str, default="results",
                        help="directory to hold result plots")
    parser.add_argument("--batch_size", type=int, default=100,
                            help="size of the mini-batch")
    parser.add_argument("--waves_file", type=str, default="waves.npy",
                        help="filename of the serialized sine data")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level, default: %(default)s)")
    parser.add_argument("--log_file", type=str, default="logs.txt",
                        help="filename for the log file for metrics")
    parser.add_argument("--override_hid_dims", default=False, action="store_true",
                        help="Override representation dimensions to `hid_dim`")
    parser.add_argument("--hid_dim", type=int, default=1024,
                        help="dimension to override representations")
    parser.add_argument("--test", default=False, action="store_true",
                        help="Test instead of train")
    parser.add_argument("--ntest_ctx", type=int, default=785,
                        help="number of test context phase values")
    parser.add_argument("--gen_test_ctx", default=False, action="store_true",
                        help="Generate and serialize context indices for test")
    parser.add_argument("--ckpt_fname", type=str,
                        help="checkpoint filename, expected in "
                             "'outdir'/logs/checkpoints")

    # Add Trainer args
    parser = Trainer.add_argparse_args(parser)
    # Add model specific args
    parser = SPSystemBase.add_model_specific_args(parser)
    args = parser.parse_args()

    # Override representations dim if needed
    if args.override_hid_dims:
        args = override_hidden_dims(args, args.hid_dim)

    # Initialize pytorch
    init_torch(args.seed)
    seed_everything(args.seed)

    # Create output directory
    outroot = Path(args.outdir)
    outroot.mkdir(parents=True, exist_ok=True)
    args.log_file = str(outroot / args.log_file)

    # Setup logging
    configure_logging(args.log_level, args.log_file)

    # Load data, set paths
    artefacts_dir = (Path(__file__).resolve().parent.parent / "artefacts/")
    dataset_dir = artefacts_dir / "datasets/synthetic/sine-waves"
    waves_path = dataset_dir / args.waves_file
    waves = np.load(waves_path)

    # Train on phases from [0, 2*pi)
    logging.info("Training")
    phases = np.arange(0, 2*np.pi, 0.001)
    train_set = SyntheticGlancing(waves, args.future_len)
    test_ctx_idxs_path = dataset_dir / "phased-test-ctx-idxs-2pi.npy"

    # Update args with values needed for loading models
    args.enc_nhid = 32
    args.no_pool = True
    args.observed_len = waves.shape[0] - args.future_len
    args.nposes = 1

    # Train
    if not args.test:
        train(train_set, outroot, args)

    # Evaluate
    if args.test:
        # Set path vars and create the output directory
        ckpt_path = outroot / args.ckpt_fname
        resdir = ckpt_path.parents[0] / args.resdir
        resdir.mkdir(parents=True, exist_ok=True)
        if args.gen_test_ctx:
            ctx_idxs = generate_test_ctx_idxs(phases, args.ntest_ctx)
            np.save(test_ctx_idxs_path, ctx_idxs)
        else:
            ctx_idxs = np.load(test_ctx_idxs_path)
        # Compute test metrics
        compute_test_metrics(train_set, ckpt_path, ctx_idxs, resdir, args)


if __name__ == "__main__":
    main()
