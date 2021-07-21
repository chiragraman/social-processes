#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: social_process.py
# Created Date: Sunday, November 17th 2019, 5:27:58 pm
# Author: Chirag Raman
#
# Copyright (c) 2019 Chirag Raman
###


from typing import  NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

import data.types as types
import models.sequential as seq
from common.tensor_op import multi_range
from models.attention import AttenderBase
from models.sequential import DeterministicEncDecType, StochasticEncDecType
from .neural_process import NeuralProcess, ZEncoder
from .pooling import SocialPooler


class Seq2SeqProcessComponents(NamedTuple):

    """ The components required by a Seq2Seq Process """

    trg_encdec: StochasticEncDecType
    latent_encdec: DeterministicEncDecType
    z_encoder: ZEncoder
    det_encdec: Optional[DeterministicEncDecType] = None
    attender: Optional[AttenderBase] = None


class SocialProcessSeq2Seq(nn.Module):

    """ Implement a seq2seq neural process """

    def __init__(
            self, components: Seq2SeqProcessComponents, norm_rot: bool = True,
            nposes: int = 1, skip_deterministic_decoding: bool = False
    ) -> None:
        """ Initialize the process """
        super().__init__()

        self.trg_encdec = components.trg_encdec
        self.latent_encdec = components.latent_encdec
        self.z_encoder = components.z_encoder
        self.det_encdec = components.det_encdec
        self.attender = components.attender
        self.norm_rot = norm_rot
        self.nposes = nposes
        self.skip_deterministic_decoding = skip_deterministic_decoding

    def _normalize_rot(
            self, mean: Tensor, std: Tensor = None
        ) -> Tuple[Tensor, Optional[Tensor]]:
        """ Normalize the rotation components of predictions

        Rotation components are expected to be of 4 dimensions (quaternions)
        with a pose being denoted by 7 dimensions (4 rot + 3 loc)

        """
        mean_normed = mean.clone()
        rot_idx = multi_range(4, self.nposes, 7)
        # Compute the norm, shape (..., self.nposes)
        norm = (mean[..., rot_idx]
                .view(*mean.size()[:-1], self.nposes, 4)
                .norm(dim=-1))
        # Repeat elements to match the rotation terms in the last dimension
        # norm is now (..., self.poses * 4)
        norm = torch.repeat_interleave(norm, 4, dim=-1)
        # Normalize the means
        mean_normed[..., rot_idx] = mean[..., rot_idx] / norm
        # Normalize the std by the same scaling factor (norm of means)
        std_normed = None
        if std is not None:
            std_normed = std.clone()
            std_normed[..., rot_idx] = std[..., rot_idx] / norm
        return mean_normed, std_normed

    def reparameterize_latent(self, r: Tensor) -> Normal:
        """ Reparameterize the aggregated representation into z

        Args:
            r   --  The aggregated representation of the context
                    (batch_size * npeople, r_dim + pooled.nout)

        Returns the q distribution parameterized by the z encoder
        q   --  torch.distributions.Normal with loc and scale of shape
                (batch_size, z_dim)

        """
        # Aggregate rep over the people in the same group
        # Since all sequences in batch are from the same group, also
        # aggregate over the batch dimension
        r_agg = r.mean(dim=0) # (r_dim + pooled.nout)
        # Get mean and variance for q(z|...)
        # mu : (1, z_dim), sigma : (1, z_dim)
        z_distrib = self.z_encoder(r_agg.unsqueeze(0))
        return z_distrib

    def _encode_latent(self, observed: Tensor) -> Normal:
        """ Encode observed sequences into a latent distribution

        Args:
            observed    --  The observed sequences
                            (seq_len, batch_size, npeople, data_dim)

        Returns the q distribution parameterized by the z encoder
        q   --  torch.distributions.Normal with loc and scale of shape
                (batch_size, z_dim)

        """
        # Encode the observed seqs, rep is (nlayers, batch_size * npeople, dim)
        rep = self.latent_encdec.encode_sequences(observed)
        rep = rep.mean(0) # (batch_size * npeople, dim)
        return self.reparameterize_latent(rep)

    def _encode_deterministic(self, context: Tensor,
                              trg_observed: Tensor = None) -> Tensor:
        """ Encode the context sequences into the deterministic representation

        Args:
            context         --  The context sequences
                                (seq_len, nsequences, npeople, data_dim)
            trg_observed    --  The target observed sequences if
                                cross-attention is used
                                (seq_len, batch_size, npeople, data_dim)

        Returns the deterministic representation of the context of shape
        (batch_size*npeople, r_dim)

        """
        s, n, p, _ = context.shape
        m = trg_observed.size(1)
        # Encode the context seqs, rep is (nlayers, n*p, dim)
        # After taking mean across layers, rep is (1, n*p, dim)
        rep = self.det_encdec.encode_sequences(context).mean(0, keepdim=True)
        ctx = context.view(s, n*p, -1).transpose(0, 1) # (n*p, s, data_dim)
        trg = trg_observed.view(s, m*p, -1).transpose(0, 1) # (m*p, s, data_dim)
        val = self.attender(trg.unsqueeze(0), ctx.unsqueeze(0), rep) # (1, m*p, dim)
        return val.squeeze(0) # (m*p, dim)

    def _decode_deterministic(
            self, enc_dec: DeterministicEncDecType,
            samples: types.Seq2SeqSamples, teacher_forcing: float = 0
        ) -> Tensor:
        """ Get deterministic predictions from the enc_dec model """
        det_futures = enc_dec(samples, teacher_forcing)
        if self.norm_rot:
            # Normalize the mean and std. rotation components of the
            # predictions. This is done to obtain a unit quaternion
            det_futures, _ = self._normalize_rot(det_futures)
        return det_futures

    def _predict(
            self, samples: types.Seq2SeqSamples, q_distrib: Normal,
            nz_samples: int, teacher_forcing: float = 0, context: Tensor = None
    ) -> Normal:
        """ Sample from the encoded distribution and make predictions

        Args:
            samples         --  The samples for which to make predictions
                                observed (seq_len, batch_size, npeople, data_dim)
            q_distrib       --  The encoded normal distribution with loc and
                                scale of shape (batch_size, z_dim)
            nz_samples      --  The number of z samples to use for estimation
            teacher_forcing --  The probability of using teacher forcing
            context         --  The context sequences to condition on
                                (seq_len, nsequences, npeople, data_dim)
                                optional, used only for deterministic path

        Returns the predicted Normal distribution with location and scale
        of shape (nz_samples, target_len, batch_size, npeople, data_dim)
        """
        # Sample a batch of z's using the reparameterization trick
        # (nz_samples, batch_size, z_dim)
        z_samples = q_distrib.rsample([nz_samples])
        z_samples = z_samples.expand(-1, samples.observed.shape[1], -1)

        # Encode latent representation r_context for the deterministic path
        r_context = None
        if self.det_encdec is not None:
            r_context = self._encode_deterministic(context, samples.observed)

        # Get the predictive distribution of targets y*
        future_mu, future_sigma = self.trg_encdec(
            samples, z_samples, r_context, teacher_forcing
        )
        # Normalize the mean and std. rotation components of the predictions
        # This is done to obtain a unit quaternion
        if self.norm_rot:
            future_mu, future_sigma = self._normalize_rot(
                future_mu, future_sigma
            )
        return Normal(future_mu, future_sigma)

    def forward(self, split: types.DataSplit, nz_samples: int = 1,
                teacher_forcing: float = 0.5) -> types.Seq2SeqPredictions:
        """ Training/Validation forward pass for the social process

        Args:
            split           --  The context and target split of data
            nz_samples      --  The number of z samples to use for estimation
            teacher_forcing --  The probability of using teacher forcing


        Returns:
            The Seq2SeqPredictions object containing the predicted and computed
            quantities.

        """
        ctx = split.context
        trg = split.target
        # Get z distribution parameters for the context and target sequences
        q_context = self._encode_latent(ctx.observed)
        q_target = None
        q = q_context
        if self.training:
            # Encode the target sequences too while training
            q_target = self._encode_latent(trg.observed)
            q = q_target

        # Get deterministic decoded futures on the latent and deterministic
        # paths to train the representations to also be informative
        # of the future directly
        latent_path_futures = None
        det_path_futures = None
        if  not self.skip_deterministic_decoding:
            latent_path_futures = self._decode_deterministic(
                self.latent_encdec, trg, teacher_forcing
            )
            if self.det_encdec is not None:
                det_path_futures = self._decode_deterministic(
                    self.det_encdec, trg, teacher_forcing
                )

        # Predict by sampling from the encoded distribution
        p_future = self._predict(trg, q, nz_samples, teacher_forcing,
                                 ctx.observed)
        return types.Seq2SeqPredictions(
            stochastic=p_future,
            posteriors=types.ApproximatePosteriors(q_context, q_target),
            deterministic=(latent_path_futures, det_path_futures)
        )

    def predict(self, samples: types.Seq2SeqSamples, context: Tensor,
                nz_samples: int = 1) -> types.Seq2SeqPredictions:
        """ Make predictions for the provided data samples

        Expected to be used at test time, does not perform deterministic
        decoding, and only encodes the context sequences for the latent path

        Args:
            samples     --  The samples to make predictions for
            context     --  The context sequences to condition on
            nz_samples  --  The number of z samples to use for estimation
        """
        q = self._encode_latent(context)
        p_future = self._predict(samples, q, nz_samples, teacher_forcing=0,
                                 context=context)
        return types.Seq2SeqPredictions(
            stochastic=p_future, posteriors=types.ApproximatePosteriors(q)
        )