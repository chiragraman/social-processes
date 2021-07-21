#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: processes.py
# Created Date: Friday, February 7th 2020, 10:42:04 am
# Author: Chirag Raman
#
# Copyright (c) 2020 Chirag Raman
###


import abc
import logging
from argparse import ArgumentParser
from argparse import Namespace
from typing import Dict, Tuple, Optional, Sequence, Union, Type

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
from torch.optim import Adam, lr_scheduler, Optimizer

import models.social_process as sp
import models.neural_process as nproc
from .builders import RecurrentBuilder, MLPBuilder
from data.types import (
    ApproximatePosteriors, DataSplit, Seq2SeqSamples, Seq2SeqPredictions
)
from models.attention import AttentionType, QKRepresentation, build_attender
from common.regularizer import OrthogonalRegularizer
from common.utils import EnumAction
from models.neural_process import ZEncoder
from train.loss import SocialProcessLoss, SocialAuxLoss


CHECKPOINT_MONITOR_METRIC = "monitored_nll"

def _add_common_args(parent_parser: ArgumentParser) -> ArgumentParser:

    """ Add args common to all neural process systems

    These include the dimension/layers, training, attention, and viz args

    """
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--nlayers", type=int, default=2,
                        help="number of recurrent/linear layers, default is 2")
    parser.add_argument("--nz_layers", type=int, default=2,
                        help="number of hidden layers in z Encoder, default is 2")
    parser.add_argument("--enc_nhid", type=int, default=64,
                        help="encoder hidden dimensions, default is 64")
    parser.add_argument("--dec_nhid", type=int, default=64,
                        help="decoder hidden dimensions, default is 64")
    parser.add_argument("--r_dim", type=int, default=64,
                        help="deterministic encoding dimension, "
                        "default is 64")
    parser.add_argument("--z_dim", type=int, default=64,
                        help="stochastic latent dimension, default is 64")
    parser.add_argument("--data_dim", type=int, default=1,
                        help="data dimension, default is 1")
    parser.add_argument("--use_deterministic_path", default=False,
                        action="store_true",
                        help="include separate path for deterministic encoding")

    # Training args
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate, default is 1e-3")
    parser.add_argument("--schedule_lr", default=False, action="store_true",
                        help="use a learning rate scheduler")
    parser.add_argument("--lr_steps", nargs="+", type=int,
                        help="epochs at which to schedule lr")
    parser.add_argument("--lr_gamma", type=float, default=0.3,
                        help="gamma factor for lr decay in MultiStepLR, "
                             "default is 0.3")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay, default is 5e-4")
    parser.add_argument("--reg", type=float, default=1e-6,
                        help="regularization constant for rnn weights, "
                        "default is 1e-6")
    parser.add_argument("--dropout", type=float, default=0,
                        help="dropout value for the network components")
    parser.add_argument("--teacher_forcing", type=float, default=0.5,
                        help="fraction of ground truth labels to use for "
                        "training, default is 0.5")

    # Attention args
    parser.add_argument("--attention_type", type=AttentionType,
                        action=EnumAction, default=AttentionType.UNIFORM,
                        help="type of deterministic path cross-attention")
    parser.add_argument("--attention_rep", type=QKRepresentation,
                        action=EnumAction, default=QKRepresentation.IDENTITY,
                        help="transformation applied to attention keys and"
                             " queries")
    parser.add_argument("--attention_qk_dim", type=int, default=32,
                        help="query and key dimension for attention; if "
                             "`attention_rep` is IDENTITY, needs to be "
                             "equal to the input dimension to the attender "
                             "module. This is handled internally for NP Models")
    parser.add_argument("--attention_nheads", type=int, default=8,
                        help="number of heads for multihead attention")

    # Logging/Visualization args
    parser.add_argument("--ndisplay", type=int, default=25,
                        help="Number of epochs after which to plot samples")
    return parser


class AbstractCommonBase(pl.LightningModule, abc.ABC):

    """ Implement common behavior across child lightning modules

    Implement manual end-of-epoch logging and configuration of optimizers

    """

    def __init__(self, hparams: Union[Namespace, Dict]) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)

    def on_epoch_start(self):
        """ Print epoch number at the start of each epoch """
        logging.info(f"[*] Epoch {self.current_epoch+1}...")

    def training_epoch_end(self, train_outs: Sequence):
        """ Print/Log training loss and metrics at the end of each epoch """
        self.shared_epoch_end(train_outs, "train")

    def validation_epoch_end(self, valid_outs: Sequence):
        """ Print/Log validation loss and metrics at the end of each epoch """
        metrics = self.shared_epoch_end(valid_outs, "valid")
        nll = metrics["nll"]
        # Log for checkpointing
        self.log(CHECKPOINT_MONITOR_METRIC, nll)

    def _log_epoch(self, mode: str, metric_tag: str, metric: Tensor) -> None:
        """ Log the metrics to tensorboard as well as logging channels """
        logging.info(f"--- Average {mode} {metric_tag}: {metric:.4f}")
        self.logger.experiment.add_scalar(
            f"epoch_{mode}/{mode}_{metric_tag}", metric,
            global_step=self.current_epoch+1
        )

    def epoch_metrics(self, outputs: Sequence) -> Dict[str, Tensor]:
        """ Compute the metrics to log at the end of the epoch

        Args:
            outputs --  The sequence of outputs from the individual steps

        Returns the dictionary of metric tag to metric value

        """
        metric_dict = dict()
        for metric_name in ["loss", "nll", "kl", "aux_loss"]:
            batch_metrics = [item.get(metric_name) for item in outputs]
            if not None in batch_metrics:
                avg_metric = torch.Tensor(batch_metrics).mean()
            metric_dict[metric_name] = avg_metric
        return metric_dict

    def shared_epoch_end(self, outputs: Sequence, mode: str) -> None:
        """ Log metrics at the end of the epochs

        Args:
            outputs --  The sequence of outputs from the individual steps
            mode    --  A label denoting the training mode

        """
        metrics = self.epoch_metrics(outputs)
        for item in metrics:
            self._log_epoch(mode, metric_tag=item, metric=metrics[item])
        return metrics

    def configure_optimizers(self):
        """ Setup optimizers for training the Social Process """
        optimizer = self.create_optimizer()
        scheduler = (lr_scheduler.MultiStepLR(
                        optimizer, milestones=self.hparams.lr_steps,
                        gamma=self.hparams.lr_gamma)
                     if self.hparams.schedule_lr else None)
        return {"optimizer": optimizer, "lr_schedulers": scheduler}

    @abc.abstractmethod
    def create_optimizer(self) -> Optimizer:
        """ Create the optimizer, called in `configure_optimizers` """
        pass


class SPSystemBase(AbstractCommonBase):

    """ System describing the base behaviour of a Social Process """

    def __init__(self, hparams: Namespace,
                 builder: Type[Union[RecurrentBuilder, MLPBuilder]]) -> None:
        """ Initialize the Social Process module.

        Args:
            hparams             -- The hyperparameters for the experiment
            builder             -- The factory to build backbone components

        """
        #### Lightning broke hparams type saving again.
        if type(hparams) != Namespace:
            hparams = Namespace(**hparams)
        ############################
        super().__init__(hparams)
        # Build components
        components = builder.init_components(hparams)
        # Initialize the process
        self.process = sp.SocialProcessSeq2Seq(
            components, not hparams.skip_normalize_rot, hparams.nposes,
            hparams.skip_deterministic_decoding
        )
        # Initialize regularizer
        self.reg = OrthogonalRegularizer(reg=hparams.reg)
        # Initialize loss module
        self.loss = self._configure_loss(hparams)

    def _configure_loss(self, hparams: Namespace) -> nn.Module:
        """ Configure the loss module """
        return SocialProcessLoss(nn.MSELoss())

    def forward(self, batch: DataSplit) -> Seq2SeqPredictions:
        """ Perform inference for the Social Process """
        preds = self.process.predict(batch.target, batch.context.observed,
                                     self.hparams.nz_samples)
        return preds

    def shared_step(
            self, batch: DataSplit, teacher_forcing: float) -> Tensor:
        """ Compute the loss for a single batch """
        preds = self.process(batch, self.hparams.nz_samples, teacher_forcing)
        return self.loss(preds, batch)

    def training_step(self, batch: DataSplit, _) -> Tensor:
        """ Perform a single step in the training loop """
        loss, nll, kl, aux = self.shared_step(batch, self.hparams.teacher_forcing)
        loss_no_reg = loss.detach().clone()
        loss += self.reg(self.process)
        logs = {"step_loss_no_reg": loss_no_reg, "step_loss_reg": loss.detach(),
                "step_nll": nll, "step_kl": kl, "step_aux_loss": aux}
        self.log_dict(logs, prog_bar=True)
        return {"loss": loss, "loss_no_reg": loss_no_reg, "nll": nll,
                "kl": kl, "aux_loss": aux}

    def validation_step(self, batch: DataSplit, _) -> None:
        """ Perform an evaluation step """
        loss, nll, kl, aux = self.shared_step(batch, teacher_forcing=0)
        return {"loss": loss.detach(), "nll": nll, "kl": kl, "aux_loss": aux}

    def test_step(self, batch: DataSplit, _) -> Seq2SeqPredictions:
        """ Perform a test step """
        predictions = self.forward(batch)
        return predictions

    def epoch_metrics(self, outputs: Sequence) -> Dict[str, Tensor]:
        """ Compute the metrics to log at the end of the epoch

        Args:
            outputs --  The sequence of outputs from the individual steps

        Returns the dictionary of metric tag to metric value

        """
        metrics = super().epoch_metrics(outputs)
        if self.training:
            avg_raw_loss = torch.Tensor(
                [item["loss_no_reg"] for item in outputs]
            ).mean()
            metrics["no_reg_loss"] = avg_raw_loss
        return metrics

    def create_optimizer(self) -> Optimizer:
        """ Create the optimizer, called in `configure_optimizers` """
        optimizer = Adam(
            list(self.process.parameters()), lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """ Add args pertaining to the model and training of the process """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--nz_samples", type=int, default=1,
                            help="number of z samples for mc estimation, only "
                            "used if `--fix_variance` is false, default is 10")
        parser.add_argument("--fix_variance", default=False, action="store_true",
                            help="Fix the predicted variance; default is to "
                            "learn the variance")
        parser.add_argument("--share_target_encoder", default=False,
                            action="store_true",
                            help="share the target and latent path encoders")
        parser.add_argument("--skip_normalize_rot", default=False, action="store_true",
                            help="skip normalization of rotation features; by "
                                 "default normalizes rotation features to "
                                 "unit quaternion")
        parser.add_argument("--skip_deterministic_decoding", default=False,
                            action="store_true",
                            help="skip deterministic decoding of futures, "
                                 "only considered if `use_deterministic_path`"
                                 " is True")
        return _add_common_args(parser)


class SPSystemSocial(SPSystemBase):

    """ System for running experiments on Social Datasets """

    def _configure_loss(self, hparams:Namespace) -> nn.Module:
        """ Configure the loss module """
        sx, sq = hparams.homo_init
        aux_loss = SocialAuxLoss(sx, sq, hparams.nposes)
        return SocialProcessLoss(aux_loss)

    def create_optimizer(self):
        """ Setup optimizers for training the Social Process """
        optimizer = Adam(
            list(self.process.parameters()) + list(self.loss.parameters()),
            lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """ Add args pertaining to the model and training of the process """
        parser = SPSystemBase.add_model_specific_args(parent_parser)
        parser.add_argument("--nposes", type=int, default=1,
                            help="number of poses per person; eg. 2 for head "
                            "and body pose. default is 1")
        parser.add_argument("--pooler_nhid_embed", type=int, default=64,
                            help="embedding dimension for spatial pooler, "
                            "default is 64")
        parser.add_argument("--pooler_nhid_pool", type=int, default=64,
                            help="hidden dimension for spatial pooler's "
                            "pre-pooling layers, default is 64")
        parser.add_argument("--pooler_nout", type=int, default=64,
                            help="output dimension for spatial pooler, "
                            "default is 64")
        parser.add_argument("--pooler_stride", type=int, default=1,
                            help="temporal stride for social pooling, "
                            "default is 1")
        parser.add_argument("--pooler_temporal_nhid", type=int, default=64,
                            help="hidden dimension for temporal pooler encoder "
                            "default is 64")
        parser.add_argument("--no-pool", default=False, action="store_true",
                            help="turn off social pooling")
        parser.add_argument("--homo_init", metavar=("sx", "sq"),
                            type=float, nargs=2, default=(0.0, -3.0),
                            help="initial guess for homoscedastic uncertainties "
                                 "variables(default: %(default)s)")
        return parser


class NPSystemBase(AbstractCommonBase):

    """ System describing the base behaviour of a Neural Process """

    def __init__(self, hparams: Namespace) -> None:
        """ Initialize the Social Process module.

        Args:
            hparams -- The hyperparameters for the experiment

        """
        super().__init__(hparams)

        # Set the dimensions
        ninp = self.hparams.data_dim * self.hparams.observed_len
        nout = self.hparams.data_dim * self.hparams.future_len

        # Initialize the attention module if needed
        attender = None
        if self.hparams.use_deterministic_path:
            attender = build_attender(
                self.hparams.attention_type, self.hparams.attention_rep,
                self.hparams.attention_qk_dim, self.hparams.r_dim, ninp,
                nheads=self.hparams.attention_nheads,
                dropout=self.hparams.dropout
            )

        # Initialize the process
        # Note that for running with the same social data, a sequence of shape
        # (seq_len, nsequences, npeople, data_dim) is reshaped into
        # (1, nsequences*npeople, seq_len*data_dim) for both context and target
        self.process = nproc.NeuralProcess(
            ninp, nout, self.hparams.r_dim, self.hparams.z_dim,
            self.hparams.enc_nhid, self.hparams.nlayers,
            self.hparams.nz_layers, self.hparams.use_deterministic_path,
            attender, self.hparams.dropout
        )
        # Initialize loss module
        self.loss = self._configure_loss()

    def _configure_loss(self) -> nn.Module:
        """ Configure the loss module """
        return SocialProcessLoss()

    def reshape_samples(
            self, samples: Seq2SeqSamples
        ) -> Tuple[Tensor, Optional[Tensor]]:
        """ Reshape the observed and future tensors from Seq2Seq Samples

        The observed and future tensors are reshaped from
        (seq_len, nsequences, npeople, data_dim) into
        (1, nsequences*npeople, seq_len*data_dim)

        Returns a tuple of (observed, future) tensors, future maybe None

        """
        so, n, p, d = samples.observed.size()
        obs = samples.observed.permute(1, 2, 0, 3).contiguous().view(n*p, so*d)
        obs = obs.unsqueeze(0)
        fut = None
        if samples.future is not None:
            sf = samples.future.size(0)
            fut = samples.future.permute(1, 2, 0, 3).contiguous().view(n*p, sf*d)
            fut = fut.unsqueeze(0)
        return obs, fut

    def reshape_batch(
            self, batch: DataSplit
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """ Reshape context and target feature tensors in the data split

        The observed and future tensors are reshaped from
        (seq_len, nsequences, npeople, data_dim) into
        (1, nsequences*npeople, seq_len*data_dim)

        """
        x_context, y_context = self.reshape_samples(batch.context)
        x_target, y_target = self.reshape_samples(batch.target)
        return (x_context, y_context, x_target, y_target)

    def forward(self, batch: DataSplit) -> Seq2SeqPredictions:
        """ Perform inference for the Social Process """
        #  Reshape context and target sequences of shape
        # (seq_len, nsequences, npeople, data_dim) into
        # (1, nsequences*npeople, seq_len*data_dim)
        batch_data = self.reshape_batch(batch)
        y_pred, q_ctx, q_trg = self.process(*batch_data)
        # Reshape predicted distribution and wrap into a Seq2SeqPredictions obj
        # y_pred tensors are finally (seq_len, nsequences, npeople, data_dim)
        _, ntrg, p, d = batch.target.observed.size()
        mu = y_pred.loc.squeeze(0).view(ntrg, p, -1, d).permute(2, 0, 1, 3)
        sig = y_pred.scale.squeeze(0).view(ntrg, p, -1, d).permute(2, 0, 1, 3)
        # Final y_pred is supposed to be (nz_samples, seq_len, ...)
        preds = Seq2SeqPredictions(
            stochastic=Normal(mu.unsqueeze(0), sig.unsqueeze(0)),
            posteriors=ApproximatePosteriors(q_ctx, q_trg)
        )
        return preds

    def shared_step(self, batch: DataSplit) -> Tensor:
        """ Compute the loss for a single batch """
        preds = self.forward(batch)
        return self.loss(preds, batch)

    def training_step(self, batch: DataSplit, _) -> Tensor:
        """ Perform a single step in the training loop """
        loss, nll, kl, _ = self.shared_step(batch)
        logs = {"step_loss": loss.detach(), "step_nll": nll, "step_kl": kl}
        self.log_dict(logs, prog_bar=True)
        return {"loss": loss, "nll": nll, "kl": kl}

    def validation_step(self, batch: DataSplit, _) -> None:
        """ Perform an evaluation step """
        loss, nll, kl, _ = self.shared_step(batch)
        return {"loss": loss.detach(), "nll": nll, "kl": kl}

    def test_step(self, batch: DataSplit, _) -> Seq2SeqPredictions:
        """ Perform a test step """
        predictions = self.forward(batch)
        return predictions

    def create_optimizer(self):
        """ Setup optimizers for training the Social Process """
        optimizer = Adam(
            list(self.process.parameters()),
            lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """ Add args pertaining to the model and training of the process """
        return _add_common_args(parent_parser)
