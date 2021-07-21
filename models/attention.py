#!/usr/bin/ed_v python3
# -*- coding:utf-8 -*-
###
# File: attention.py
# Created Date: Monday, November 30th 2020, 10:50:13 am
# Author: Chirag Raman
#
# Copyright (c) 2020 Chirag Raman
###


import abc
import logging
import math
from enum import Enum, auto
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from common.architecture.mlp import MLP
from common.initialization import initialize_params


class AttentionType(Enum):

    """ Enum for denoting attention type """

    UNIFORM     = auto()
    DOT         = auto()
    MULTIHEAD   = auto()


class QKRepresentation(Enum):

    """ Representation for keys and queries """

    IDENTITY    = auto()
    MLP         = auto()
    RNN         = auto()


class AttenderBase(abc.ABC, nn.Module):

    """
    Implement the Attention mechanism for a Neural Process

    """

    def __init__(
            self, rep: QKRepresentation = QKRepresentation.IDENTITY,
            d_qk: Optional[int] = None, d_v: Optional[int] = None,
            ninp: Optional[int] = None, dropout: float = 0
        ) -> None:
        """ Initialize the attender """
        super().__init__()
        self.d_qk = d_qk
        self.d_v = d_v
        self.rep = rep
        if rep == QKRepresentation.MLP:
            self.qk_rep = MLP(ninp, d_qk, d_qk, dropout=dropout,
                              act_kwargs={"inplace":"True"})
        elif rep == QKRepresentation.RNN:
            self.qk_rep = nn.GRU(ninp, d_qk, batch_first=True)
            self.qk_rep.apply(initialize_params)
        elif rep == QKRepresentation.IDENTITY:
            self.qk_rep = nn.Identity()
            assert d_qk == ninp, (
                "If representation is identity, query/key dimension should "
                "match input dimension; got d_qk = {}, ninp = {}".format(d_qk, ninp))

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """ Compute the value for the query after attending to the keys.

        If rep is KQRepresentation.IDENTITY, expects input dimension to match
        d_qk.

        Args:
            q   --  queries. tensor of shape (B,m,ninp) if not a sequence, else
                    (B, m, seq_len, ninp)
            k   --  queries. tensor of shape (B,n,ninp) if not a sequence, else
                    (B, n, seq_len, ninp)
            v   --  values. tensor of shape (B,n,d_v)

        Returns:
            tensor of shape (B,m,d_v)

        """
        b = q.size(0)
        if self.rep == QKRepresentation.RNN:
            q = q.view(-1, q.size(2), q.size(3)) # (B*m, seq_len, ninp)
            k = k.view(-1, k.size(2), k.size(3)) # (B*n, seq_len, ninp)

        # Transform the queries and keys
        # (B, <m or n>, d_qk) or (B, <m or n>, seq_len, d_qk) if idnetity rep or
        # (output, h_n) where h_n (1, B*<m or n>, d_qk) if RNN rep
        q = self.qk_rep(q)
        k = self.qk_rep(k)

        if self.rep == QKRepresentation.RNN:
            # Use only h_n as the representation for the sequences,
            q = q[1].squeeze(0).view(b, -1, self.d_qk) # (B, m, d_qk)
            k = k[1].squeeze(0).view(b, -1, self.d_qk) # (B, n, d_qk)

        return self.attend(q, k, v)

    @abc.abstractmethod
    def attend(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        pass


class UniformAttender(AttenderBase):

    """ Compute mean of values as an aggregation """

    def attend(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """ Compute mean of values as an aggregation, equivalent to np

        Args:
            q   --  queries. tensor of shape (B,m,d_qk) or (B,m,s,d_qk)
            k   --  keys. tensor of shape (B,n,d_qk) or (B,n,s,d_qk)
            v   --  values. tensor of shape (B,n,d_v)

        Returns:
            tensor of shape (B,m,d_v)

        """
        npoints = q.size(1)
        rep = torch.mean(v, dim=1, keepdim=True)
        rep = rep.repeat(1, npoints, 1)
        return rep


class DotAttender(AttenderBase):

    """ Compute scaled dot attention """

    def attend(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """ Computes dot product attention with scaling.

        Args:
            q   --  queries. tensor of  shape (B,m,d_qk) or (B,m,s,d_qk)
            k   --  keys. tensor of shape (B,n,d_qk) or (B,n,s,d_qk)
            v   --  values. tensor of shape (B,n,d_v)

        Returns:
            tensor of shape (B,m,d_v)

        """
        scale = math.sqrt(q.size(-1))

        q_shape = "bmsk" if len(q.size()) == 4 else "bmk"
        k_shape = "bnsk" if len(k.size()) == 4 else "bnk"

        unnorm_weights = torch.einsum(
            "{},{}->bmn".format(k_shape, q_shape), k, q
        )
        unnorm_weights /= scale # (B,m,n)
        weights = F.softmax(unnorm_weights, dim=-1)
        rep = torch.einsum("bmn,bnv->bmv", weights, v) # (B,m,d_v)
        return rep


class MultiheadAttender(AttenderBase):

    """ Compute Multihead attention """

    def __init__(
            self, rep: QKRepresentation, d_qk: int, d_v: int,
            ninp: int, dropout:float = 0, nheads: int = 8
        ) -> None:
        """ Initialize the attender and matrices """
        super().__init__(rep, d_qk, d_v, ninp, dropout)
        self.nheads = nheads
        # Initialize the heads
        self.q_transform = nn.Linear(d_qk, d_qk * nheads, bias=False)
        self.k_transform = nn.Linear(d_qk, d_qk * nheads, bias=False)
        self.v_transform = nn.Linear(d_v, d_v * nheads, bias=False)
        nn.init.normal_(self.q_transform.weight, std=math.sqrt(1/d_qk))
        nn.init.normal_(self.k_transform.weight, std=math.sqrt(1/d_qk))
        nn.init.normal_(self.v_transform.weight, std=math.sqrt(1/d_v))
        self.unify_heads = nn.Linear(nheads * d_v, d_v)

    def attend(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """ Compute multi-head attention.

        Args:
            q       --  queries. tensor of  shape (B,m,d_qk)
            k       --  keys. tensor of shape (B,n,d_qk)
            v       --  values. tensor of shape (B,n,d_v)

        Returns:
            tensor of shape (B,m,d_v)

        """
        b, m, _ = q.size()
        n = k.size(1)
        h = self.nheads

        queries = self.q_transform(q).view(b, m, h, self.d_qk)
        keys = self.k_transform(k).view(b, n, h, self.d_qk)
        values = self.v_transform(v).view(b, n, h, self.d_v)

        # Compute the dot product
        dot = torch.einsum("bmhk,bnhk->bhmn", queries, keys) / math.sqrt(self.d_qk)
        dot = F.softmax(dot, dim=-1)

        # Compute the h outputs
        out = torch.einsum("bhmn,bnhv->bmhv", dot, values)

        # Unify the heads
        out = self.unify_heads(torch.reshape(out, (b, m, h*self.d_v)))

        return out


def build_attender(
        attention_type: AttentionType, rep: QKRepresentation,
        d_qk: Optional[int] = None, d_v: Optional[int] = None,
        ninp: Optional[int] = None, dropout: float = 0, **kwargs
    ) -> AttenderBase:
    """ Initialize an attender according to the given hyperparams """
    # Ensure d_qk matches ninp
    if rep == QKRepresentation.IDENTITY and d_qk != ninp:
        logging.info("[!] Setting d_qk equal to input dim for attender")
        d_qk = ninp
    if attention_type == AttentionType.UNIFORM:
        attender = UniformAttender(rep, d_qk, d_v, ninp, dropout)
    elif attention_type == AttentionType.DOT:
        attender = DotAttender(rep, d_qk, d_v, ninp, dropout)
    elif attention_type == AttentionType.MULTIHEAD:
        attender = MultiheadAttender(rep, d_qk, d_v, ninp, dropout, **kwargs)
    return attender
