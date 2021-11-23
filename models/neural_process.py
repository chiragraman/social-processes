#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: neural_process.py
# Created Date: Wednesday, May 6th 2020, 4:24:20 pm
# Author: Chirag Raman
#
# Copyright (c) 2020 Chirag Raman
###


from typing import Tuple


import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
from torch.nn import functional as F

from common.architecture.mlp import MLP
from models.attention import AttenderBase

class NPEncoder(nn.Module):

    """
    Map an input, output (x_i, y_i) pair to a representation r_i.

    Attributes:
        ninp    -- Dimension of inputs/features.
        nout    -- Dimension of outputs.
        nhid    -- Dimension of hidden layer.
        nrep    -- Dimension of the output representation r.
        nlayers -- Number of hidden layers.

    """

    def __init__(self, ninp: int, nout: int, nhid: int,
                 nrep: int, nlayers: int = 3, dropout: float = 0) -> None:
        """ Initialize the encoder. """
        super().__init__()
        self.input_to_hidden = MLP(
            ninp+nout, nrep, nhid, nlayers, dropout=dropout,
            act_kwargs={"inplace":"True"}
        )


    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Encode the input-output pairs into representations.

        Args:
            x   -- features (batch_size, npoints, ninp)
            y   -- outputs (batch_size, npoints, nout)

        Returns:
            representation r (batch_size, npoints, nrep)

        """
        # Flatten tensors
        batch_size, npoints, _ = x.size()
        x_flat = x.view(batch_size * npoints, -1)
        y_flat = y.contiguous().view(batch_size * npoints, -1)
        input_pairs = torch.cat((x_flat, y_flat), dim=1)
        return self.input_to_hidden(input_pairs).view(batch_size, npoints, -1)


class ZEncoder(nn.Module):

    """
    Encode the representation r into a the latent distribution z.

    The representation r is used to parameterize the normally distributed
    latent encoding z through the reparameterization trick.

    Attributes:
        ninp    -- Dimension of the r representation.
        nout    -- Dimension of z representation.
        nhid    -- Dimension of hidden layer.
        nlayers -- Number of hidden layers.

    """

    def __init__(self, ninp: int, nout: int, nhid: int = 16,
                 nlayers: int = 3, dropout: float = 0) -> None:
        """ Initialize the ZEncoder """
        super().__init__()
        # Intermediate fully connected layers
        self.ninp = ninp
        self.nout = nout
        self.shared_layers = MLP(ninp, nhid, nhid, nlayers, dropout=dropout,
                                 act_kwargs={"inplace":"True"})
        # Pass shared representation through specialised 2-layer MLPs for
        # computing sufficient statistics of the latent distribution
        self.zmean = MLP(nhid, nout, nhid, nlayers=2, dropout=dropout,
                         act_kwargs={"inplace":"True"})
        self.zstd = MLP(nhid, nout, nhid, nlayers=2, dropout=dropout,
                        act_kwargs={"inplace":"True"})

    def forward(self, r: Tensor) -> Normal:
        """ Reparameterize r into z.

        Args:
            r   --  Encoded representation of the input-output pairs,
                    shape (batch_size, ninp)

        Returns:
            The Normal latent distrubution for z, with parameters of shape
            (batch_size, nout)

        """
        shared_rep = self.shared_layers(r)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives"
        mu = self.zmean(shared_rep)
        sigma = 0.1 + 0.9 * torch.sigmoid(self.zstd(shared_rep))
        return Normal(mu, sigma)


class NPDecoder(nn.Module):

    """
    Map target inputs x*, r, and z samples from q(z|context) to predictions y*.

    Attributes:
        ninp    -- Dimension of inputs/features x*.
        nrep    -- Dimension of representation r.
        nlatent -- Dimension of latent representation z.
        nout    -- Dimension of predictions y*
        nhid    -- Dimension of hidden layer.
        nlayers -- Number of hidden layers

    """

    def __init__(self, ninp: int, nout: int, nhid: int, nlayers: int = 3,
                 dropout: float = 0) -> None:
        """ Initialize the encoder. """
        super().__init__()

        self.shared_layers = MLP(ninp, nhid, nhid, nlayers, dropout=dropout,
                                 act_kwargs={"inplace":"True"})
        # Pass shared representation through specialised 2-layer MLPs for
        # computing sufficient statistics of the prediction distribution
        self.ymean = MLP(nhid, nout, nhid, dropout=dropout,
                         act_kwargs={"inplace":"True"})
        self.ystd = MLP(nhid, nout, nhid, dropout=dropout,
                        act_kwargs={"inplace":"True"})

    def forward(self, x: Tensor, z: Tensor, r: Tensor = None) -> Normal:
        """
        Decode the x*, r*, and z tuples into predictions y*.

        The decoder accommodates deterministic and stochastic paths by
        accepting deterministic representations r and stochastic samples z.

        Args:
            x   -- features (batch_size, npoints, ninp)
            z   -- stochastic path samples (batch_size, nlatent)
            r   -- deterministic representations (batch_size, npoints, nrep)

        Returns:
            The Normal distribution for the prediction, with parameters of
            shape (batch_size, npoints, nout)

        """
        batch_size, npoints, _ = x.size()
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, nlatent) to (batch_size, npoints, nlatent)
        z = z.unsqueeze(1).repeat(1, npoints, 1)
        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * npoints, -1)
        z_flat = z.view(batch_size * npoints, -1)
        if r is not None:
            r_flat = r.view(batch_size * npoints, -1)
            z_flat = torch.cat([z_flat, r_flat], dim=-1)
        # Input is concatenation of z with every row of x
        input_cat = torch.cat((x_flat, z_flat), dim=-1)
        hidden = self.shared_layers(input_cat)
        mu = self.ymean(hidden)
        pre_sigma = self.ystd(hidden)
        # Reshape output into expected shape
        mu = mu.view(batch_size, npoints, -1)
        pre_sigma = pre_sigma.view(batch_size, npoints, -1)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        return Normal(mu, sigma)


class NeuralProcess(nn.Module):

    """ Implement a Neural Process model .

    Attributes:
        ninp    -- Dimension of inputs/features x.
        nout    -- Dimension of predictions y.
        nrep    -- Dimension of representation r.
        nlatent -- Dimension of latent representation z.
        nhid    -- Dimension of hidden layers for component networks.
        nlayers -- Number of hidden layers for component networks

    """

    def __init__(
            self, ninp: int, nout: int, nrep: int, nlatent: int, nhid: int,
            nlayers: int = 3, nz_layers: int = 2,
            use_deterministic_path: bool = False,
            attender: AttenderBase = None, dropout: float = 0
        ) -> None:
        """ Initialise the process. """
        super().__init__()

        self.stochastic_encoder = NPEncoder(ninp, nout, nhid, nrep,
                                            nlayers, dropout)
        z_nhid = min(nrep, nlatent) + abs(nlatent - nrep) // 2
        self.z_encoder = ZEncoder(nrep, nlatent, z_nhid, nlayers=nz_layers,
                                  dropout=dropout)
        self.attender = attender
        self.use_deterministic_path = use_deterministic_path
        if use_deterministic_path:
            self.deterministic_encoder = NPEncoder(ninp, nout, nhid, nrep,
                                                   nlayers, dropout)
            dec_ninp = ninp + nrep + nlatent
        else:
            self.deterministic_encoder = None
            dec_ninp = ninp + nlatent
        # dec_nhid = min(dec_ninp, nout) + abs(nout - dec_ninp) // 2
        dec_nhid = nhid
        self.decoder = NPDecoder(dec_ninp, nout, dec_nhid, nlayers, dropout)

    def _encode_latent(self, x: Tensor, y: Tensor) -> Normal:
        """ Map the input-output pair into the latent representations.

        This is a helper function to map (x, y) pairs into the Normal
        distribution z using the reparameterization trick.

        Args:
            x   --  inputs, shape (batch_size, npoints, ninp)
            y   --  outputs, shape (batch_size, npoints, nout)

        Returns:
            q   --  Normal distribution for z with parameters of shape
                    (batch_size, nlatent)

        """
        r = self.stochastic_encoder(x, y)
        # Aggregate r
        r = r.mean(dim=1)
        z_distrib = self.z_encoder(r)
        return z_distrib

    def _encode_deterministic(self, x_context: Tensor, y_context: Tensor,
                              x_target: Tensor) -> Tensor:
        """ Encode the context pairs into the deterministic representation

        Args:
            x_context   --  context inputs, shape (batch_size, nctx, ninp)
            y_context   --  context outputs, shape (batch_size, nctx, nout)
            x_target    --  target inputs, shape (batch_size, ntrg, ninp)

        Returns:
            r   --  deterministic rep of shape shape
                    (batch_size, ntrg, nrep)

        """
        rep = self.deterministic_encoder(x_context, y_context)
        val = self.attender(x_target, x_context, rep)
        return val

    def forward(self, x_context: Tensor, y_context: Tensor, x_target: Tensor,
                y_target: Tensor = None) -> Tuple[Tensor, Normal, Normal]:
        """
        Forward pass through the neural process to return a distribution over
        the target points y_target.

        Args:
            x_context   --  Context inputs, shape (batch_size, ncontext, ninp).
                            Note that x_context is a subset of x_target.
            y_context   --  Context outputs, shape (batch_size, ncontext, nout)
            x_target    --  Target inputs, shape (batch_size, ntarget, ninp)
            y_target    --  Optional, target outputs, used during training,
                            shape (batch_size, ntarget, nout)

        Returns a tuple of:
            y_pred      --  The normal distribution over the target y with
                            parameters of shape (batch_size, ntarget, nout)
            q_context   --  The latent distribution over context points
            q_target    --  The latent distribution over target points

        """
        # Encode the context
        q_context = self._encode_latent(x_context, y_context)
        q = q_context
        q_target = None
        if self.training:
            # Encode the target as well
            q_target = self._encode_latent(x_target, y_target)
            q = q_target
        # Sample from the encoded latent distribution
        z_sample = q.rsample()
        # Encode the deterministic representation
        det_r = None
        if self.use_deterministic_path:
            det_r = self._encode_deterministic(x_context, y_context, x_target)
        # Predict the target distribution based on context
        y_pred = self.decoder(x_target, z_sample, det_r)
        return (y_pred, q_context, q_target)