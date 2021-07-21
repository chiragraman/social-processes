#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: types.py
# Created Date: Thursday, February 13th 2020, 4:39:01 pm
# Author: Chirag Raman
#
# Copyright (c) 2020 Chirag Raman
###


from enum import Enum, auto
from typing import (
    Any, Dict, NamedTuple, Optional, Sequence, Tuple, Union
)

from torch import Tensor
from torch.distributions import Normal


class ApproximatePosteriors(NamedTuple):

    """ Represents the approixmate posterior distributions q(z|...)

    Attributes:
        q_context       -- The approixmate posterior distribution computed
                           over the context points
        q_target        -- The approixmate posterior distribution computed
                           over the target points

    """
    q_context: Normal
    q_target: Normal = None


class Seq2SeqSamples(NamedTuple):

    """ Represents a seq2seq data sample

    Attributes:
        key             --  An identifier for the group to which the sample belongs
        observed_start  --  The frame at which the observed sequence starts
        observed        --  The observed sequence features
        future_len      --  The number of time steps to predict in the future
        offset          --  The time offset between the end of the observed and
                            sequence and start of the future sequence
        future          --  The predicted sequence features

    """
    key: Any
    observed_start: int
    observed: Tensor
    future_len: int
    offset: int = 1
    future: Tensor = None


class Seq2SeqPredictions(NamedTuple):

    """ Represents a prediction from the Seq2Seq Social Process

    Attributes:
        stochastic      -- The predicted output distribution from the
                           stochastic encoder-decoder
        posteriors      -- The approixmate posterior distribution computed
                           over the context and target points
        deterministic   -- The deterministic encoder-decoder predictions for
                           the latent and deterministic paths

    """
    stochastic: Normal
    posteriors: ApproximatePosteriors
    deterministic: Tuple[Optional[Tensor], Optional[Tensor]] = (None, None)


class DataSplit(NamedTuple):

    """ Represents the context and target split of data

    Attributes:
        context -- The context points to condition on
        target  -- The target points to make predictions for

    """
    context: Any
    target: Any


class ModelType(Enum):

    """ Enum for denoting type of model to train """

    SOCIAL_PROCESS  = auto()
    NEURAL_PROCESS  = auto()


class ComponentType(Enum):

    """ Module for summarizing sequence of embeddings """

    MLP = auto()
    RNN = auto()


class ContextRegime(Enum):

    """ Choice of using fixed context or random context """

    FIXED   = auto()
    RANDOM  = auto()


"""
Map for specifying which individual sequences to serialize at test.
Maps group id to ranges of start frames of observed sequences
"""
SerializationMap = Dict[str, Union[int, Sequence[Tuple[int]]]]