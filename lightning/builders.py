#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: builders.py
# Created Date: Friday, January 8th 2021, 2:58:38 pm
# Author: Chirag Raman
#
# Copyright (c) 2021 Chirag Raman
###


from argparse import Namespace
from typing import Optional, Tuple, Type, Union

import torch.nn as nn

from common.architecture.mlp import MLP
from common.initialization import initialize_params
from data.types import ComponentType
from models.attention import build_attender, AttenderBase
from models.neural_process import ZEncoder
from models.pooling import SocialPooler
from models.sequential import (
    EncoderRNN, DecoderRNN, DeterministicEncoderDecoderRNN,
    StochasticEncoderDecoderRNN, TemporalPooler, EncoderMLP,
    DeterministicEncoderDecoderMLP, StochasticEncoderDecoderMLP,
    StochasticEncDecType, DeterministicEncDecType, EncoderType, DecoderType
)
from models.social_process import Seq2SeqProcessComponents


def init_social_pooling(hparams: Namespace) -> TemporalPooler:
    """ Initialize the spatial and temporal social poolers """
    pooler = None
    if not hparams.no_pool:
        social_pooler = SocialPooler(
            hparams.data_dim, hparams.pooler_nhid_embed, hparams.pooler_nhid_pool,
            hparams.r_dim, hparams.pooler_nout, hparams.nposes,
            dropout=hparams.dropout
        )
        pooler_enc = None
        if hparams.pooler_stride != -1:
            if hparams.component == ComponentType.RNN:
                pooler_enc = EncoderRNN(
                    hparams.pooler_nout, hparams.pooler_temporal_nhid,
                    hparams.pooler_nout, hparams.nlayers,
                    hparams.dropout
                )
                pooler_enc.apply(initialize_params)
            elif hparams.component == ComponentType.MLP:
                pooler_enc = EncoderMLP(
                    hparams.pooler_nout, hparams.pooler_temporal_nhid,
                    hparams.observed_len, hparams.pooler_temporal_nhid,
                    hparams.nlayers, hparams.dropout
                )
        pooler = TemporalPooler(
            social_pooler, hparams.pooler_stride, pooler_enc
        )
    return pooler


def init_attender(hparams: Namespace) -> AttenderBase:
    """ Initialize the attender module for cross-attention """
    attender = build_attender(
        hparams.attention_type, hparams.attention_rep, hparams.attention_qk_dim,
        hparams.r_dim, hparams.data_dim, nheads=hparams.attention_nheads,
        dropout=hparams.dropout
    )
    return attender


def init_process_components(
        stochastic_type: StochasticEncDecType,
        deterministic_type: DeterministicEncDecType,
        hparams: Namespace, encoder: EncoderType, decoder: DecoderType,
        latent_encoder: EncoderType, latent_decoder: DecoderType,
        det_encoder: Optional[EncoderType] = None,
        det_decoder: Optional[DecoderType] = None,
        pooler: Optional[TemporalPooler] = None, **enc_dec_kwargs: int
    ) -> Seq2SeqProcessComponents:
    """ Initialize the encoder-decoder, attender, and ZEncoder modules """
    latent_seq2seq = deterministic_type(latent_encoder, latent_decoder,
                                        pooler=pooler)
    det_seq2seq = None
    attender = None
    encoded_rep_dim = hparams.r_dim + hparams.z_dim

    if hparams.use_deterministic_path:
        det_seq2seq = deterministic_type(
            det_encoder, decoder=det_decoder, pooler=pooler
        )
        attender = init_attender(hparams)
        encoded_rep_dim += hparams.r_dim # Account for r_context

    z_nhid = (min(hparams.r_dim, hparams.z_dim)
              + abs(hparams.r_dim - hparams.z_dim) // 2)
    z_enc = ZEncoder(hparams.r_dim, hparams.z_dim, nhid=z_nhid,
                     nlayers=hparams.nz_layers, dropout=hparams.dropout)

    if stochastic_type == StochasticEncoderDecoderMLP:
        trg_seq2seq = stochastic_type(encoder, decoder, pooler=pooler)
    elif stochastic_type == StochasticEncoderDecoderRNN:
        trg_seq2seq = stochastic_type(
            encoder, decoder, encoded_rep_dim,
            pooler=pooler, fix_variance=hparams.fix_variance
        )
    else:
        raise ValueError("Unexpected stochastic encoder-decoder type")
    return Seq2SeqProcessComponents(
        trg_seq2seq, latent_seq2seq, z_enc, det_seq2seq, attender
    )


class RecurrentBuilder:

    """ Build the recurrent components for the Social Process """

    @classmethod
    def init_components(cls, hparams: Namespace) -> Seq2SeqProcessComponents:
        """ Initialize the seq2seq and z encoder models """
        # Initialize sequence encoders and decoders
        enc, dec, lat_enc, lat_dec, det_enc, det_dec = cls.init_rnn(
            hparams.data_dim, hparams.enc_nhid, hparams.r_dim, hparams.dec_nhid,
            hparams.z_dim, hparams.nlayers, hparams.fix_variance,
            hparams.share_target_encoder, hparams.use_deterministic_path,
            hparams.skip_deterministic_decoding, hparams.dropout
        )

        # Initialize the pooler if needed
        pooler = init_social_pooling(hparams)

        # Initialize the encoder-decoders and attender
        components = init_process_components(
            StochasticEncoderDecoderRNN, DeterministicEncoderDecoderRNN,
            hparams, enc, dec, lat_enc, lat_dec, det_enc, det_dec, pooler
        )
        cls.init_component_weights(
            components.trg_encdec, components.latent_encdec,
            components.det_encdec
        )
        return components

    @staticmethod
    def init_component_weights(
            stoch_seq2seq: StochasticEncoderDecoderRNN,
            latent_seq2seq: DeterministicEncoderDecoderRNN,
            det_seq2seq: DeterministicEncoderDecoderRNN = None
        ) -> None:
        """ Initialize the weights in the models """
        stoch_seq2seq.apply(initialize_params)
        latent_seq2seq.apply(initialize_params)
        if det_seq2seq is not None:
            det_seq2seq.apply(initialize_params)

    @staticmethod
    def init_rnn(
            data_dim: int, encoder_nhid: int, encoder_nout: int,
            decoder_nhid: int, z_dim: int, nlayers: int, fix_variance: bool,
            share_trg_encoder: bool = True, deterministic_path: bool = False,
            skip_det_decoding: bool = False, dropout: float = 0
        ) -> Tuple[EncoderRNN, DecoderRNN, EncoderRNN, DecoderRNN, EncoderRNN,
                   DecoderRNN]:
        """ Initialise the rnn component encoders and decoders

        Args:
            hparams :   The hyper parameters for initializing the component
                        networks
        """
        # Target encoder
        encoder = EncoderRNN(data_dim, encoder_nhid, encoder_nout, nlayers,
                             dropout)

        # Latent encoder decoder
        if not share_trg_encoder:
            lat_encoder = EncoderRNN(data_dim, encoder_nhid, encoder_nout,
                                     nlayers, dropout)
        else:
            lat_encoder = encoder
        lat_decoder = None
        if not skip_det_decoding:
            lat_decoder = DecoderRNN(data_dim, decoder_nhid, data_dim,
                                     nlayers=nlayers, dropout=dropout)

        # Deterministic encoder decoder
        det_encoder = None
        det_decoder = None
        if deterministic_path:
            det_encoder = EncoderRNN(data_dim, encoder_nhid, encoder_nout,
                                     nlayers, dropout)
            if not skip_det_decoding:
                det_decoder = DecoderRNN(data_dim, decoder_nhid, data_dim,
                                         nlayers=nlayers, dropout=dropout)
        if not fix_variance:
            data_dim *= 2

        # Target decoder
        decoder = DecoderRNN(
            data_dim, decoder_nhid, data_dim, nlayers, dropout
        )
        return (encoder, decoder, lat_encoder, lat_decoder,
                det_encoder, det_decoder)


class MLPBuilder:

    """ Build the MLP components for the Social Process """

    @classmethod
    def init_components(cls, hparams: Namespace) -> Seq2SeqProcessComponents:
        """ Initialize the seq2seq and z encoder modules """
        # Initialize sequence encoders
        encoder, lat_encoder, det_encoder = cls.init_seq_encoders(
            hparams.data_dim, hparams.r_dim, hparams.observed_len,
            hparams.r_dim, hparams.nlayers, hparams.dropout,
            hparams.use_deterministic_path, hparams.share_target_encoder
        )

        # Initialize the sequence decoders
        decoder, lat_decoder, det_decoder = cls.init_seq_decoders(
            hparams.data_dim, hparams.future_len, hparams.r_dim,
            hparams.z_dim, hparams.nlayers, hparams.dropout,
            hparams.use_deterministic_path, hparams.skip_deterministic_decoding
        )

        # Initialize the pooler if needed
        pooler = init_social_pooling(hparams)

        # Initialize the encoder-decoders and attender
        components = init_process_components(
            StochasticEncoderDecoderMLP, DeterministicEncoderDecoderMLP,
            hparams, encoder, decoder, lat_encoder, lat_decoder, det_encoder,
            det_decoder, pooler
        )
        return components

    @staticmethod
    def init_seq_encoders(
            data_dim: int, r_dim: int, seq_len: int, nhid: int, nlayers: int,
            dropout: float = 0, deterministic_path: bool = False,
            share_encoder: bool = True
        ) -> Tuple[EncoderMLP, Optional[EncoderMLP], Optional[EncoderMLP]]:
        """ Initialize the Encoder MLPs """
        # Initialize encoders
        trg_encoder = EncoderMLP(
            data_dim, r_dim, seq_len, nhid, nlayers, dropout=dropout
        )
        if not share_encoder:
            latent_encoder = EncoderMLP(
                data_dim, r_dim, seq_len, nhid, nlayers, dropout=dropout
            )
        else:
            latent_encoder = trg_encoder
        det_encoder = None
        if deterministic_path:
            det_encoder = EncoderMLP(data_dim, r_dim, seq_len, nhid,
                                     nlayers, dropout=dropout)

        return trg_encoder, latent_encoder, det_encoder

    @staticmethod
    def init_seq_decoders(
            data_dim: int, seq_len: int, r_dim: int, z_dim: int,
            nlayers: int, dropout: float = 0, deterministic_path: bool = False,
            skip_det_decoding: bool = False
        ) -> Tuple[MLP, Optional[MLP], Optional[MLP]]:
        """ Initialize the Encoder MLPs """
        # Initialize the latent deterministic decoder if needed
        lat_decoder = None
        if not skip_det_decoding:
            lat_decoder = MLP(
                r_dim, seq_len * data_dim, r_dim, nlayers=1, dropout=dropout,
                act_kwargs={"inplace":"True"}
            )

        # Initialize the deterministic path decoder if needed
        ninp = r_dim + z_dim
        det_decoder = None
        if deterministic_path:
            # Expand dimensions to include r_context
            ninp += r_dim
            if not skip_det_decoding:
                # Initialize the det_decoder
                det_decoder = MLP(
                    r_dim, seq_len * data_dim, r_dim, nlayers=1, dropout=dropout,
                    act_kwargs={"inplace":"True"}
                )
        # Initialize the stochastic decoder
        trg_decoder = MLP(ninp, seq_len * data_dim * 2, ninp, nlayers,
                          dropout=dropout, act_kwargs={"inplace":"True"})

        return trg_decoder, lat_decoder, det_decoder
