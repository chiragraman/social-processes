#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: encoder_rnn.py
# Created Date: Thursday, November 7th 2019, 4:27:00 pm
# Author: Chirag Raman
#
# Copyright (c) 2019 Chirag Raman
###


import math
from typing import Callable, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from common.architecture.mlp import MLP
from data.types import Seq2SeqSamples
from .pooling import SocialPooler


class EncoderRNN(nn.Module):

    """Encode sequence inputs into representations, r_i."""

    def __init__(self, ninp: int, nhid: int, nout: int, nlayers: int = 1,
                 dropout: float = 0) -> None:
        """ Initialize the model """
        super().__init__()

        self.gru = nn.GRU(ninp, nhid, nlayers, dropout=dropout)
        self.out_linear = nn.Linear(nhid, nout)
        self.hid_linear = nn.Linear(nhid, nout)
        self.nhid = nhid
        self.nout = nout
        self.nlayers = nlayers

    def forward(self, inputs: Tensor) -> Tensor:
        """ Forward pass to encode the sequence.

        Returns output (seq_len, batch_size, self.nout),
                h_n (1, batch_size, self.nout)
        """
        output, h_n = self.gru(inputs)
        output = self.out_linear(output)
        # Assuming uni-directional, so last dim is nhid
        h_n = self.hid_linear(h_n)
        return output, h_n

    def init_hidden(self, batch_size):
        """ Initialise the hidden tensors """
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, batch_size, self.nhid)


class DecoderRNN(nn.Module):

    """ Decode the latent representation into a predicted sequence """

    def __init__(self, ninp: int, nhid: int, nout: int, nlayers: int = 1,
                 dropout: float = 0) -> None:
        """ Initialize the model """
        super().__init__()

        # Input -- previous prediction and context tensor
        self.gru = nn.GRU(ninp + nhid, nhid, nlayers, dropout=dropout)
        # Input -- previous prediction, context vector, and RNN output as input
        self.out = nn.Linear(ninp + nhid * 2, nout)

        self.nhid = nhid
        self.nlayers = nlayers
        self.nout = nout

    def forward(self, inputs: Tensor, hidden: Tensor, context: Tensor) -> Tensor:
        """ Forward pass to decode into a target sequence

        Args:
            inputs  -- tensor (batch_size, ninp)
            hidden  -- tensor (nlayers, batch_size, nhid)
            context -- tensor (batch_size, nhid)

        Returns decoded sequence and last hidden state of the gru
        sequence    -- tensor (batch_size, nout)
        hidden      -- tensor (nlayers, batch_size, nhid)
        """
        # Concatenate inputs & contexts resulting in a
        # (1, batch_size, ninp + nhid) dim tensor
        inputs = inputs.unsqueeze(0)
        inputs = torch.cat((inputs, context.unsqueeze(0)), dim=2)

        # Typically,
        # output -- (seq_len, batch_size, nhid * n_directions)
        # hidden -- (n_layers * n_directions, batch_size, nhid)
        #
        # For the decoder, seq_len = n_directions = n_layers = 1, since the
        # decoder is invoked one timestep at a time. This results in -
        #
        # output -- (1, batch_size, nhid)
        # h_n    -- (nlayers, batch_size, nhid)
        output, h_n = self.gru(inputs, hidden)

        # Allow the linear layer to see the context tensor, the input, as well
        # as the output to generate the final output.
        # output -- (batch size, ninp + nhid * 2)
        output = torch.cat((inputs.squeeze(0), output.squeeze(0)), dim=1)

        # Generate final prediction
        # prediction -- (batch size, nout)
        prediction = self.out(output)

        return prediction, h_n

    def init_hidden(self, batch_size):
        """ Initialise the hidden tensors """
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, batch_size, self.nhid)


class EncoderMLP(nn.Module):

    """Encode sequence inputs into representations using an MLP"""

    def __init__(self, ninp: int, nout: int, seq_len: int, nhid: int,
                 nlayers: int = 1, dropout: float = 0) -> None:
        """ Initialize the model """
        super().__init__()

        self.mlp = MLP(
            seq_len * ninp, seq_len * nout, nhid, nlayers, dropout=dropout,
            act_kwargs={"inplace":"True"}
        )
        self.summarizer = MLP(
            seq_len * nout, nout, nout, nlayers=1, dropout=dropout,
            act_kwargs={"inplace":"True"}
        )
        self.nout = nout

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """ Forward pass to encode the sequence.

        Args:
            inputs (seq_len, batch_size, data_dim)

        Returns embeddings per timestep (seq_len, batch_size, self.nout)
                summary (1, batch_size, self.nout)

        """
        s, b, d = inputs.size()
        # (s, batch, d) -> (batch, s * d)
        inputs = inputs.permute(1, 0, 2).contiguous().view(-1, s*d)
        # (batch, s * nout)
        embeddings = self.mlp(inputs)
        # summarized (1, batch, nout)
        summary = self.summarizer(embeddings).unsqueeze(0)
        return embeddings.view(b, s, -1).permute(1, 0, 2).contiguous(), summary


class TemporalPooler(nn.Module):

    """ Encode temporal dynamics of social context in a conversation.

    Refer to `models.pooling.SocialPooler` for details on how social context is
    encoded at a single time step. This module models the temporal dynamics of
    how social context evolves over time by performing pooling at a specific
    sampling rate and processing it using an RNN encoder. Using a separate
    EncoderRNN to perform temporal modeling allows for the pooling sampling
    rate to be independent of the sampling rate of the raw features.

    Attributes:
        encoder     --  The RNN or MLP to model the temporal dynamics in the
                        social context, optional if stride = -1
        nout        --  The dimension of the output tensor, which matches
                        encoder.nhid
        pooler      --  The SocialPooler object for performing pooling at the
                        individual time steps
        stride      --  The time step interval at which to apply pooling;
                        following values are accepted: an integer between
                        [1, seq_len-1] for pooling at the corresponding
                        stride, -1 for pooling at the last time step of the
                        sequence (note that pytorch, unlike numpy, does not
                        support negative stride for inverse ordering),
                        default=1

    """

    def __init__(
            self, pooler: SocialPooler, stride: int = 1,
            encoder: Optional[Union[EncoderRNN, EncoderMLP]] = None
        ) -> None:
        """ Initialize the temporal pooler object """
        super().__init__()
        if stride != -1:
            assert (encoder is not None), (
                "Temporal pooler needs an encoder if stride is not -1"
            )
        self.pooler = pooler
        self.stride = stride
        self.encoder = encoder
        self.nout = self.encoder.nout if encoder is not None else pooler.nout

    def forward(self, features: Tensor, embeddings: Tensor) -> Tensor:
        """ Forward pass through the temporal pooling module

        Encode social context from each individual's perspective. Refer
        SocialPooler for details.

        Args:
            features    --  The input features of people in the scene
                            shape (seq_len, batch_size, npeople, data_dim)
            embeddings  --  The embeddings of the raw features for the sequence
                             shape (seq_len, batch_size, npeople, embedding_dim)

        Returns
            The pooled tensor for the people in the scene, which is the h_n
            from the encoder. If an RNN encoder, has shape
            (self.encoder.nlayers, batch_size * npeople, self.encoder.nout), or
            (1, batch_size * npeople, self.encoder.nout)

        """
        # Validate stride values
        assert (self.stride == -1 or 1 <= self.stride <= features.size(0)-1), (
            "stride value should be -1 or in range [1, seq_len-1] inclusive"
        )
        # Sample every `stride` time steps along the seq_len dim or take the
        # last time step if stride is -1
        seq = features[::self.stride] if self.stride != -1 else features[-1:]
        emb = embeddings[::self.stride] if self.stride != -1 else embeddings[-1:]
        # Flatten seq_len and batch dims
        s, b, p, _ = seq.size()
        seq = seq.view(s*b, p, -1)
        emb = emb.view(s*b, p, -1)
        # Pool
        pooled = self.pooler(seq, emb)
        # Convert pooled tensor back to expected shape. If stride is -1, s = 1
        pooled = pooled.view(s, b*p, -1)
        if self.stride != -1:
            # (1 / nlayers, b*p, -1)
            _, pooled = self.encoder(pooled)
        return pooled


class FutureOffsetEncoder(nn.Module):

    def __init__(self, nembed: int, dropout: float = 0.25, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        if nembed % 2 != 0:
            raise ValueError("Cannot use sinusoidal offset encoding with "
                             "odd embedding dim (for dim={:d})".format(nembed))
        oe = torch.zeros(max_len, nembed)
        offsets = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, nembed, 2).float()\
                   * (-math.log(10000.0) / nembed))
        oe[:, 0::2] = torch.sin(offsets * div_term)
        oe[:, 1::2] = torch.cos(offsets * div_term)
        self.register_buffer("oe", oe)

    def forward(self, r: Tensor, offset: Tensor, npeople: int) -> Tensor:
        """ Add to the representation the encoding for the specific offset

        Args:
            r       --  The representation of the sequence
                        (nlayers, batch_size * npeople, nembed)
            offset  --  Refer `data.types.Seq2SeqSamples.offset`
                        (batch_size)
            npeople --  The number of people in a group

        """
        offsets = torch.repeat_interleave(offset, npeople)
        r += self.oe[offsets]
        return self.dropout(r)


class SocialSeq2SeqBase(nn.Module):

    """ Provide common encoding behavior for Social Seq2Seq models """

    def __init__(
            self, encoder: Union[EncoderRNN, EncoderMLP],
            pooler: Optional[TemporalPooler] = None
        ) -> None:
        """ Initialize the object

        Args:
            encoder             --  The encoder RNN or MLP
            pooler              --  The temporal pooling module for encoding
                                    social context for each person in a
                                    conversing group. Ensure that the
                                    encoder.nlayers is the same as
                                    pooler.encoder.nlayers, optional

        """
        super().__init__()

        self.encoder = encoder
        self.pooler = pooler
        self.pooled_projector = None
        if pooler is not None:
            self.pooled_projector = nn.Linear(encoder.nout + pooler.nout, encoder.nout)
        self.offset_encoder = FutureOffsetEncoder(encoder.nout)

    def encode_sequences(self, obs: Tensor) -> Tensor:
        """ Encode observed sequences with temporal social pooling

        Args:
            obs --  The features for the observed sequence,
                    (seq_len, batch_size * npeople, data_dim)

        Returns the encoded and optionally pooled representation
        (nlayers, batch_size * npeople, encoder.nout (+ pooler.nout))

        """
        s, b, p, d = obs.size()
        flat_obs = obs.view(s, b*p, d)

        # Encode source sequence into deterministic representation
        # ts_emb (seq_len, batch_size * npeople, encoder.nout)
        # encoded (1/self.encoder.nlayers, batch_size * npeople, encoder.nout)
        ts_emb, encoded = self.encoder(flat_obs)

        if self.pooler is not None:
            # Perform pooling over relative partner features to encode social
            # context for each conversing group
            # pooled (pooler.encoder.nlayers, batch_size * npeople, pooler.nout)
            pooled = self.pooler(obs, ts_emb.view(s, b, p, -1))

            # Concatenate pooled representation with encoded rep
            # encoded (1/nlayers, batch_size * npeople, encoder.nout + pooler.nout)
            encoded = torch.cat((encoded, pooled), dim=-1)

            # Reproject concatenated representation to encoder output dim
            # (1/nlayers, batch_size * npeople, encoder.nout)
            encoded = self.pooled_projector(encoded)

        return encoded

    def forward(self, samples: Seq2SeqSamples) -> Tuple[Tensor, Tensor, Tensor]:
        """ Encode observed sequences and prepare decoding

        Args:
            samples         --  The encapsulated observed and future pairs with
                                observed (seq_len, batch_size, npeople, data_dim)
                                future (future_len, batch_size, npeople, data_dim)

        Returns the encoded representation, starting decoding token, and
        optionally, flattened future sequence for teacher forcing
            rep         --  encoded representation of the input
                            tensor (nlayers, batch_size * npeople, encoder.nout)
            decode_start--  last time step of the observed sequence
                            tensor (batch_size * npeople, data_dim)
            flat_future --  tensor (future_len, batch_size * npeople, data_dim)
                            or None

        """
        _, b, p, d = samples.observed.size()
        fut = None
        if samples.future is not None:
            fut = samples.future.view(samples.future_len, b*p, d)

        # Encode the observed sequences with temporal social pooling
        rep = self.encode_sequences(samples.observed)

        # Add the encoding for time offset between observed and the future
        # rep is then (nlayers, batch_size * npeople, encoder.nout)
        # samples.offset is a tensor of length batch_size
        rep = self.offset_encoder(rep, samples.offset, p)

        # First input to the decoder is the last observed time step
        # (batch_size * npeople, data_dim)
        decode_start = samples.observed[-1].view(b*p, d)

        return rep, decode_start, fut


def _validate_recurrent_args(
        encoder: EncoderRNN, pooler: Optional[TemporalPooler] = None
    ) -> None:
    """ Validate dimensions of components to recurrent modules """
    if pooler is not None and pooler.stride != -1:
        assert (pooler.encoder.nlayers == encoder.nlayers), (
            "Number of encoder layers and pooler's encoder layers should "
            "match. Got pooler encoder nlayers ({}), encoder nlayers ({}) "
            .format(pooler.encoder.nlayers, encoder.nlayers)
        )


class DeterministicEncoderDecoderRNN(SocialSeq2SeqBase):

    """ Make deterministic predictions of observed sequences.

    Calling forward on this module is unsupported if a decoder is not provided.
    The decoder is optional so that deterministic decoding can be skipped.

    """

    def __init__(
            self, encoder: EncoderRNN, decoder: Optional[DecoderRNN] = None,
            pooler: Optional[TemporalPooler] = None
        ) -> None:
        """ Initialize the object

        Args:
            encoder     --  The encoder RNN
            decoder     --  The decoder RNN if deterministic decoding is
                            required
            pooler      --  The temporal pooling module for encoding social
                            context for each person in a conversing group.
                            Ensure that the encoder.nlayers is the same as
                            pooler.encoder.nlayers, optional
        """
        super().__init__(encoder, pooler)
        _validate_recurrent_args(encoder, pooler)
        self.decoder = decoder
        # Upsample encoded rep to match decoder nhid
        self.decoder_nhid_projector = None
        if decoder is not None:
            self.decoder_nhid_projector = nn.Linear(encoder.nout, decoder.nhid)

    def forward(self, samples: Seq2SeqSamples, teacher_forcing=0.5) -> Tensor:
        """ Generate future sequences for the target sequences.

        The observed sequences are encoded into a representation that is
        used for decoding into future sequences.

        Args:
            samples         --  The encapsulated observed and future pairs with
                                observed (seq_len, batch_size, npeople, data_dim)
                                future (future_len, batch_size, npeople, data_dim)
            teacher_forcing --  The probability of using teacher forcing.
                                For eg. if value is 0.75, ground truth values
                                are used 75% of the time

        Returns the predicted sequences and last hidden state r of encoder
        sequences   --  tensor (future_len, batch_size, npeople, decoder.nout)

        """
        # Encode and prepare for decoding
        _, b, p, d = samples.observed.size()
        hidden, inputs, flat_fut = super().forward(samples)

        # Upsample hidden to match decoder nhid
        # hidden is currently (nlayers, batch_size * npeople, encoder.nout)
        # After upsampling (nlayers, batch_size * npeople, decoder.nhid)
        hidden = self.decoder_nhid_projector(hidden)

        # Take mean across the layers
        # context is then (batch_size * npeople, decoder.nhid)
        context = hidden.mean(0)

        # Tensor to store generated sequence
        outseqs = torch.zeros(
            samples.future_len, b*p, self.decoder.nout
        ).to(next(self.decoder.parameters()).device)

        # Decode target sequence
        for timestep in range(0, samples.future_len):
            output, hidden = self.decoder(inputs, hidden, context)
            outseqs[timestep] = output
            inputs = output
            # Teacher forcing and set next input
            teacher_force = torch.rand(1) < teacher_forcing
            if teacher_force and flat_fut is not None:
                # If teacher forcing, use ground truth or predicted output
                inputs = flat_fut[timestep]

        return outseqs.view(samples.future_len, b, p, -1)


def combine_rz(
        z: Tensor, npeople: int, r_context: Optional[Tensor] = None
    ) -> Tensor:
    """ Combine the latent sample and r_context from a deterministic path

    Args:
        z           --  The latent samples (nsamples, batch_size, z_dim)
        npeople     --  The number of people in the sequence
        r_context   --  The representation over the context
                        (batch_size, r_dim)

    Returns concatenated latent sample
    (nsamples, batch_size * npeople, z_dim + (r_dim))

    """
    # Expand z to repeat samples for every person in the group
    # (nsamples, batch_size * npeople, z_dim)
    z = z.repeat(1, npeople, 1)

    # If conditioning directly on context, expand r_context to match z
    if r_context is not None:
        r_context = r_context.unsqueeze(0).expand(z.size(0), *r_context.size())
        z = torch.cat([z, r_context], dim=-1)

    return z


class StochasticEncoderDecoderRNN(SocialSeq2SeqBase):

    """Generate a stochastic output sequence for an observed sequence.

    The observed sequence x* and a sample from the encoding z are used to generate
    a sequence y*.

    """

    def __init__(
            self, encoder: EncoderRNN, decoder: DecoderRNN,
            encoded_rep_dim: int, pooler: Optional[TemporalPooler] = None,
            fix_variance: bool = False
        ) -> None:
        """ Initialize the object

        Args:
            encoder         --  The encoder RNN
            decoder         --  The decoder RNN
            encoded_rep_dim --  Expected dimension of
                                [encoder.nout + z_dim (+ r_context)]
            pooler          --  The temporal pooling module for encoding social
                                context for each person in a conversing group.
                                Ensure that the encoder.nlayers is the same as
                                pooler.encoder.nlayers, optional
            fix_variance    --  Learn the output variance if False
        """
        super().__init__(encoder, pooler)
        _validate_recurrent_args(encoder, pooler)
        # Project [e, r_context, z] to match decoder.nhid
        self.decoder_nhid_projector = nn.Linear(encoded_rep_dim, decoder.nhid)
        self.decoder = decoder
        self.fix_variance = fix_variance


    def _forward_decode(
            self, e, decode_start, z, future_len, future, teacher_forcing):
        """ Compute mean and sigma of future distribution for the sampled z

        Args:
            e               --  The deterministic representation of the obs seqs
                                (encoder.nlayers, batch_size * npeople,
                                 encoder.nout)
            decode_start    --  The initial token to start decoding,
                                the last timestep of the input sequences
                                (batch_size * npeople, data_dim)
            z               --  The samples from the latent space
                                (n_samples, batch_size * npeople, z_dim)
            future_len      --  The length of sequence to predict
            future          --  The decoded predicted sequences
                                (seq_len, batch_size * npeople, decoder.nout)
            teacher_forcing --  The probability of using teacher forcing.
                                For eg. if value is 0.75, ground truth values
                                are used 75% of the time

        Returns the future mean and variance
        future mean    -- tensor (n_samples, future_len, batch_size * npeople,
                                  data_dim)
        future sigma   -- tensor (n_samples, future_len, batch_size * npeople,
                                  data_dim)

        """
        # Expand r to match the number of z samples
        # shape (nsamples, encoder.nlayers, batch*npeople, encoder.nout)
        e_reshaped = e.unsqueeze(0).expand(z.size(0), *e.size())

        # Expand z to repeat for each of the nlayers
        # shape (nsamples, encoder.nlayers, batch*npeople, zdim (+ r_ctx dim))
        z_reshaped = z.unsqueeze(1).expand(z.size(0), e.size(0), *z.size()[1:])

        # ez shape (n_samples, encoder.nlayers, batch_size*npeople,
        #           encoder.nout (+ r_context dim) + z_dim)
        ez = torch.cat([e_reshaped, z_reshaped], dim=-1)

        # List to store generated futures
        futures = []

        # If variance is to be learned, repeat the input along the data dim
        data_dim = decode_start.shape[1]
        if not self.fix_variance:
            decode_start = decode_start.repeat(1, 2)

        # Iterate over ez_samples to generate a sequence for each
        for ez_sample in ez.split(1):
            # Set initial value of decoder to the concatenated latent
            # representation (nlayers, batch_size*npeople, ...)
            hidden = ez_sample.squeeze(0)

            # Reproject hidden to match decoder.nhid
            hidden = self.decoder_nhid_projector(hidden)

            # context (batch_size*npeople, ...)
            context = hidden.mean(0)

            # First input to the decoder is the last time step of the input
            # sequence (batch_size, ninp) or (batch_size, ninp*2) depending
            # on whether variance is fixed or not
            inputs = decode_start

            # Output tensor to hold generated sequence
            # (future_len, batch_size*npeople, decoder.nout)
            outseqs = torch.zeros(
                future_len, ez.size(2), self.decoder.nout
            ).to(next(self.decoder.parameters()).device)

            # Decode future sequence
            for timestep in range(0, future_len):
                output, hidden = self.decoder(inputs, hidden, context)
                if not self.fix_variance:
                    # Define sigma following convention in "Empirical E
                    # Evaluation of Neural Process Objectives"
                    std = 0.1 + 0.9*F.softplus(output[:, data_dim:])
                    output = torch.cat((output[:, :data_dim], std), dim=1)
                outseqs[timestep] = output # (batch_size, decoder.nout)
                inputs = output
                # Teacher force the inputs for the means -
                if torch.rand(1) < teacher_forcing and future is not None:
                    inputs = torch.cat(
                        (future[timestep], inputs[:, data_dim:]),
                        dim=1
                    )

            # Store generated sequences
            futures.append(outseqs)

        # Stack predictions to get a tensor of shape
        # (n_samples, future_len, batch_size*npeople, decoder.nout)
        futures = torch.stack(futures)
        mu = futures[..., :data_dim]
        sigma = futures[..., data_dim:] if not self.fix_variance \
            else torch.full(mu.size(), 0.05)

        return mu, sigma

    def forward(
            self, samples: Seq2SeqSamples, z: Tensor,
            r_context: Tensor = None, teacher_forcing: float = 0.5
    ) -> Tuple[Tensor, Tensor]:
        """ Generate future sequences for the target sequences

        Args:
            samples         --  The encapsulated observed and future pairs with
                                observed (seq_len, batch_size, npeople, data_dim)
                                future (future_len, batch_size, npeople, data_dim)
            z               --  The samples from the latent space. Note that the
                                dimensions should match as follows
                                decoder.nhid = encoder.nhid + z.shape[2]
                                (nsamples, batch_size, z_dim)
            teacher_forcing --  The probability of using teacher forcing.
                                For eg. if value is 0.75, ground truth values
                                are used 75% of the time
            r_context       --  The representation of the context set for
                                the deterministic path through the process
                                (batch_size * npeople, r_dim)

        Returns the future mean and variance
        future mean    -- tensor (n_samples, samples.future_len, batch_size,
                                  npeople, data_dim)
        future sigma   -- tensor (n_samples, samples.future_len, batch_size,
                                  npeople, data_dim)

        """
        # Encode and prepare for decoding
        _, b, p, d = samples.observed.size()
        rep, decode_start, flat_fut = super().forward(samples)

        # Combine the z and r_context tensors
        z = combine_rz(z, p, r_context)

        # Decode the future sequences
        future_mu, future_sigma = self._forward_decode(
            rep, decode_start, z, samples.future_len, flat_fut, teacher_forcing
        )

        return future_mu.view(future_mu.shape[0], samples.future_len,
                              b, p, -1),\
               future_sigma.view(future_sigma.shape[0], samples.future_len,
                                 b, p, -1)


class DeterministicEncoderDecoderMLP(SocialSeq2SeqBase):

    """ Encode the sequences into a deterministic representation r

    This module uses an MLP backbone instead of recurrent components.
    Calling forward on this module is unsupported if a decoder is not provided.
    The decoder is optional so that deterministic decoding can be skipped.

    """

    def __init__(
            self, encoder: EncoderMLP, decoder: Optional[MLP] = None,
            pooler: Optional[TemporalPooler] = None
        ) -> None:
        """ Initialize the object

        Args:
            encoder     --  The encoder MLP
            decoder     --  The decoder MLP if deterministic decoding is
                            required. Expected to accept inputs of dim
                            encoder.nout and return outputs of dim
                            future_len * data_dim
            pooler      --  The temporal pooling module for encoding social
                            context for each person in a conversing group.

        """
        super().__init__(encoder, pooler)
        self.decoder = decoder

    def forward(self, samples: Seq2SeqSamples, *args) -> Tensor:
        """ Forward pass through the encoder and decoder

        Args:
            samples         --  The encapsulated observed and future pairs with
                                observed (seq_len, batch_size, npeople, data_dim)
                                future (future_len, batch_size, npeople, data_dim)

        Returns the predicted sequences of shape
        (future_len, batch_size, npeople, data_dim)

        """
        # Encode observed samples
        # encoded rep is (1, batch_size * npeople, encoder.nout)
        _, b, p, d = samples.observed.size()
        encoded_rep, _, _ = super().forward(samples)
        # decoded is expected to be (batch_size * npeople, future_len * data_dim)
        decoded = self.decoder(encoded_rep.squeeze(0))
        decoded = (decoded.view(b, p, samples.future_len, d)
                   .permute(2, 0, 1, 3).contiguous())
        return decoded


class StochasticEncoderDecoderMLP(SocialSeq2SeqBase):

    """ Make stochastic predictions of observed sequences

    This module uses an MLP backbone instead of recurrent components.
    Refer `StochasticEncoderDecoderRNN` for the RNN variant.

    """

    def __init__(
            self, encoder: EncoderMLP, decoder: MLP = None,
            pooler: Optional[TemporalPooler] = None
        ) -> None:
        """ Initialize the object

        Args:
            encoder     --  The encoder MLP
            decoder     --  The decoder MLP. Expected to accept inputs of dim
                            (encoder.nout + z_dim + (optionally) r_context)
                            and return outputs of dim
                            (future_len * data_dim * 2) for means and variances
            pooler      --  The temporal pooling module for encoding social
                            context for each person in a conversing group.

        """
        super().__init__(encoder, pooler)
        self.decoder = decoder

    def forward(
            self, samples: Seq2SeqSamples, z: Tensor,
            r_context: Optional[Tensor] = None, *args
        ) -> Tuple[Tensor, Tensor]:
        """ Forward pass through the encoder and decoder

        Args:
            samples         --  The encapsulated observed and future pairs with
                                observed (seq_len, batch_size, npeople, data_dim)
                                future (future_len, batch_size, npeople, data_dim)
            z               --  The samples from the latent space. Note that the
                                dimensions should match as follows
                                (nsamples, batch_size, z_dim)
            r_context       --  The representation of the context set for
                                the deterministic path through the process
                                (batch_size * npeople, r_dim)

        Returns the predicted means and std sequences of shape
        (nsamples, future_len, batch_size, npeople, data_dim)

        """
        # Encode observed samples
        # encoded rep is (1, batch_size * npeople, encoder.nout)
        _, b, p, d = samples.observed.size()
        encoded_rep, _, _ = super().forward(samples)

        # Combine the z and r_context tensors
        # (nsamples, batch_size * npeople, z_dim + (r_c dim))
        z = combine_rz(z, p, r_context)

        # Combine the encoded representation of the observed sequence
        # with the latent representation
        # (nsamples, batch_size * npeople, z_dim + (r_c dim) + encoder.nout)
        rep = torch.cat((z, encoded_rep.expand(z.size(0), -1, -1)), dim=-1)

        # decoded is expected to be of shape
        # (nsamples, batch_size * npeople, future_len * data_dim * 2)
        decoded = self.decoder(rep)

         # (nsamples, future_len, batch_size, npeople, data_dim * 2)
        decoded = (decoded.view(decoded.size(0), b, p, samples.future_len, d*2)
                   .permute(0, 3, 1, 2, 4).contiguous())

        # Separate into future means and std
        mu = decoded[..., :d]
        sigma = 0.1 + 0.9*F.softplus(decoded[..., d:])

        return mu, sigma


# Type aliases for encoder-decoder components
StochasticEncDecType = Type[Union[StochasticEncoderDecoderMLP,
                                  StochasticEncoderDecoderRNN]]
DeterministicEncDecType = Type[Union[DeterministicEncoderDecoderMLP,
                                     DeterministicEncoderDecoderRNN]]
EncoderType = Union[EncoderMLP, EncoderRNN]
DecoderType = Union[DecoderRNN, MLP]
