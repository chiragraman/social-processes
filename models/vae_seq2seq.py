#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: vae_seq2seq.py
# Created Date: Thursday, November 4th 2021, 11:28:49 am
# Author: Chirag Raman
#
# Copyright (c) 2021 Chirag Raman
###


import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from common.architecture.mlp import MLP
from .neural_process import ZEncoder
from .sequential import EncoderRNN, DecoderRNN


class VariationalEncDecRNN(nn.Module):

    """ Vanilla Seq2Seq Variational Encoder Decoder with RNN encoder-decoder

    Adapted from "Generating Sentences from a Continuous Space" - Bowman et al.
    and bears similarity to "A Neural Representation of Sketch Drawings" - Ha and Eck.

    """

    def __init__(
            self, ninp: int, nout: int, nrep: int, nlatent: int, nenc_hid: int,
            ndec_hid: int, nlayers: int = 1, nz_layers: int = 2, dropout: float = 0
    ) -> None:
        """ Initalize the model

        Args:
            ninp        -- Dimension of inputs/features x.
            nout        -- Dimension of predictions y. Must match output features.
            nrep        -- Dimension of representation r.
            nlatent     -- Dimension of latent representation z.
            nenc_hid    -- Dimension of hidden layers for the encoder network.
            ndec_hid    -- Dimension of hidden layers for the decoder networks.
            nlayers     -- Number of hidden layers for component networks

        """
        super().__init__()
        assert (ninp == nout), ("VariationalEncDecRNN: Input and output feature dimensions must match")
        encoder = EncoderRNN(ninp, nenc_hid, nrep, nlayers, dropout=dropout)
        z_nhid = min(nrep, nlatent) + abs(nlatent - nrep) // 2
        z_encoder = ZEncoder(nrep, nlatent, z_nhid, nlayers=nz_layers, dropout=dropout)
        self.decoder_nhid_projector = nn.Linear(nlatent, ndec_hid)
        decoder = DecoderRNN(nout*2, ndec_hid, nout*2, nlayers, dropout=dropout)
        self.encoder = encoder
        self.z_encoder = z_encoder
        self.decoder = decoder

    def forward(self, x: Tensor, decode_len: int, y: Tensor = None,
                teacher_forcing: float = 0.5) -> Tensor:
        """ Forward pass through the variational enc-dec model

        Args:
            x               --  Sequence of input features
                                (seq_len, batch_size, encoder.ninp)
            decode_len      --  The number of timesteps to decode
            y               --  The ground truth target sequence of features
                                (seq_len, batch_size, encoder.ninp)
            teacher_forcing --  The probability of using teacher forcing.
                                For eg. if value is 0.75, ground truth values
                                are used 75% of the time

        Returns decoded mean, std, and encoded representation
        mean        -- tensor (decode_len, batch_size, encoder.ninp)
        std         -- tensor (decode_len, batch_size, encoder.ninp)
        q_z         -- Normal (batch_size, z_encoder.nout)
        encoded_rep -- tensor (batch_size, encoder.nout)
        """
        # Encode the inputs
        _, encoded = self.encoder(x) # (nlayers, batch_size, nrep)
        encoded = encoded.mean(0) # (batch_size, nrep)
        # Get the posterior distribution
        q_z = self.z_encoder(encoded.squeeze(0)) # (batch_size, nlatent)
        # Sample from the distribution
        z_sample = q_z.rsample() # (batch_size, nlatent)
        # Reproject to decoder nhid
        z_sample = self.decoder_nhid_projector(z_sample) # (batch_size, nhid)
        # Decode into sequence
        hidden = torch.tanh(z_sample)
        # (nlayers, batch_size, nlatent)
        hidden = hidden.unsqueeze(0).expand(self.decoder.nlayers, *hidden.size())

        # Output tensor to hold generated sequence
        # (decode_len, batch_size, decoder.nout)
        outseqs = torch.zeros(
            decode_len, x.size(1), self.decoder.nout
        ).to(next(self.decoder.parameters()).device)

        # Decode future sequence
        data_dim = x.size(-1)
        inputs = x[-1].repeat(1, 2) # repeat the input along the data dim to learn std
        for timestep in range(0, decode_len):
            output, hidden = self.decoder(inputs, hidden, z_sample)
            # Define sigma following convention in "Empirical E
            # Evaluation of Neural Process Objectives"
            std = 0.1 + 0.9*F.softplus(output[:, data_dim:])
            output = torch.cat((output[:, :data_dim], std), dim=1)
            outseqs[timestep] = output # (batch_size, decoder.nout)
            inputs = output
            # Teacher force the inputs for the means -
            if torch.rand(1) < teacher_forcing and y is not None:
                inputs = torch.cat((y[timestep], inputs[:, data_dim:]), dim=1)

        mu = outseqs[..., :data_dim]
        sigma = outseqs[..., data_dim:]
        return mu, sigma, q_z, encoded


class VariationalEncDecMLP(nn.Module):

    """ Vanilla Variational Encoder Decoder model with MLP encoder and decoder

    Similar to `VariationalSeq2SeqRNN`. See that class for details

    """

    def __init__(
            self, ninp: int, nout: int, nrep: int, nlatent: int, nenc_hid: int,
            ndec_hid: int, nlayers: int = 3, nz_layers: int = 2, dropout: float = 0
    ) -> None:
        """ Initalize the model

        Args:
            ninp        -- Dimension of inputs/features x.
            nout        -- Dimension of predictions y.
            nrep        -- Dimension of representation r.
            nlatent     -- Dimension of latent representation z.
            nenc_hid    -- Dimension of hidden layers for the encoder network.
            ndec_hid    -- Dimension of hidden layers for the decoder networks.
            nlayers     -- Number of hidden layers for component networks

        """
        super().__init__()

        self.encoder = MLP(ninp, nrep, nenc_hid, nlayers, dropout=dropout,
                           act_kwargs={"inplace":"True"})
        z_nhid = min(nrep, nlatent) + abs(nlatent - nrep) // 2
        self.z_encoder = ZEncoder(nrep, nlatent, z_nhid, nlayers=nz_layers,
                                  dropout=dropout)
        self.decoder_shared_layers = MLP(
            nlatent, ndec_hid, ndec_hid, nlayers, dropout=dropout,
            act_kwargs={"inplace":"True"}
        )
        # Pass shared representation through specialised 2-layer MLPs for
        # computing sufficient statistics of the prediction distribution
        self.ymean = MLP(ndec_hid, nout, ndec_hid, dropout=dropout, act_kwargs={"inplace":"True"})
        self.ystd = MLP(ndec_hid, nout, ndec_hid, dropout=dropout, act_kwargs={"inplace":"True"})

    def forward(self, x: Tensor, *args) -> Tensor:
        """ Forward pass through the variational enc-dec model

        Args:
            x               --  Sequence of input features (batch_size, ninp)

        Returns decoded mean, std, and encoded representation
        mean        -- tensor (batch_size, ninp)
        std         -- tensor (batch_size, ninp)
        q_z         -- Normal (batch_size, nlatent)
        encoded_rep -- tensor (batch_size, nrep)
        """
        # Encode the inputs
        encoded = self.encoder(x) # (batch_size, nrep)
        # Get the posterior distribution
        q_z = self.z_encoder(encoded) # (batch_size, nlatent)
        # Sample from the distribution
        z_sample = q_z.rsample() # (batch_size, nlatent)
        # Decode latent into output
        hidden = self.decoder_shared_layers(z_sample) # (batch_size, nhid)
        mu = self.ymean(hidden) # (batch_size, ninp)
        pre_sigma = self.ystd(hidden) # (batch_size, ninp)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)

        return mu, sigma, q_z, encoded
