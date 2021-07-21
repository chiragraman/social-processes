#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: pooling.py
# Created Date: Wednesday, May 13th 2020, 3:06:06 pm
# Author: Chirag Raman
#
# Copyright (c) 2020 Chirag Raman
###


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from common.architecture.mlp import MLP
from common.tensor_op import multi_range


def invert_quat(q: Tensor) -> Tensor:
    """ Invert a quaternion

    Args:
        q    --  Qauternions in scalar first format, shape (..., 4)

    Returns the inverted rotations in shape (..., 4)

    """
    inv = q.clone()
    norm = inv.norm(dim=-1)
    zero_norms = norm == 0
    if zero_norms.any():
        raise ValueError("Found zero norm quaternions in input")
    inv[..., 0] *= -1
    inv /= (norm**2).unsqueeze(-1)
    return inv


def compose_quat(q1: Tensor, q2: Tensor) -> Tensor:
    """ Compute the Hamiltonian product of q1 and q2

    Args:
        q1    --  Qauternions in scalar first format, shape (..., 4)
        q2    --  Qauternions in scalar first format, shape (..., 4)

    """
    product = torch.empty_like(q1)
    product[..., 0] = (q1[..., 0] * q2[..., 0]
                       - torch.sum(q1[..., 1:4] * q2[..., 1:4], dim=-1))
    product[..., 1:4] = (q1[..., None, 0] * q2[..., 1:4]
                         + q2[..., None, 0] * q1[..., 1:4]
                         + torch.cross(q1[..., 1:4], q2[..., 1:4]))
    product = F.normalize(product, p=2, dim=-1)
    return product


class SocialPooler(nn.Module):

    """ Encode social context from each individual's perspective.

    This is done by generating a local-global representation for a person's
    features. For incorporating the effect that conversation partners have on a
    person, this module accepts features of a group of people in conversation,
    and computes a representation for each person based on the relative
    head poses, location, and speaking status of the person's partners.

    Attributes:
        ninp        --  the input features of each person. Expected to contain
                        poses, locations, and speaking status; last
                        dimension is the speaking status.
        nhid_embed  --  the number of hidden nodes in the embedding layer
        nhid_pool   --  the number of hidden nodes in the pre-pooling layers
        nrep        --  the dimension of the embedded representation for the
                        person's features; the representations is concatenated
                        with the embedding of relative features of the partners
                        before pooling.
        nout        --  the dimension of the pooled representation for a person
        nposes      --  number of poses per person; eg. 2 for head and body
                        pose. default is 1

    """

    def __init__(self, ninp: int, nhid_embed: int, nhid_pool: int, nrep: int,
                 nout: int, nposes: int = 1, dropout: float = 0) -> None:
        """ Initialize the pooling module. """
        super().__init__()
        self.embedder = MLP(ninp, nhid_embed, nhid_embed, nlayers=2,
                            dropout=dropout, act_kwargs={"inplace":"True"})
        self.pre_pooler = MLP(nhid_embed+nrep, nout, nhid_pool, nlayers=2,
                              dropout=dropout, act_kwargs={"inplace":"True"})
        self.nout = nout
        self.nposes = nposes

    @staticmethod
    def compute_partners(features: Tensor) -> Tensor:
        """ For each person computes the features of the partners.

        For a (ngroups x npeople x features) dim input, creates a new dimension
        to contain the features of the partners, resulting in a
        (ngroups x npeople x npeople-1 x features) tensor.

        Args:
            features    --  features tensor for people in the scene.
                            (batch_size(i.e. ngroups), npeople, features)

        Returns:
            Tensor containing features of partners for each person
            (batch_size, npeople, npeople-1, features)

        """
        return torch.stack(
            [torch.cat([features[:, :idx], features[:, idx+1:]], dim=1)
             for idx in range(features.shape[1])]
        ).transpose(0, 1).contiguous()

    def relative_rotation(self, partners: Tensor, reference: Tensor) -> Tensor:
        """ Compute relative orientation of conversation partners

        Refer `relative_features` for details on how the relative
        rotation is calculated.

        Args:
            partners    --  The orientations of partners as Euler angles around
                            world coordinate axes
                            (batch_size, npeople, npeople-1, 4) where
                            npeople denotes the number of people in the group
            reference   --  The orientations of the reference persons as Euler
                            angles around world coordinate axes, repeated to
                            match the shape of the partners tensor
                            (batch_size, npeople, npeople-1, 4)

        Returns:
            The tensor of Quaternion rotations in scalar first (w, x, y, z)
            format, which when applied the partner orientation results in the
            reference orientation.

        """
        return compose_quat(reference, invert_quat(partners))

    def relative_features(self, features: Tensor) -> Tensor:
        """ Compute the realtive pose, position, and speaking status

        Let features be denoted  as rotation r [qw, qx, qy, qz],
        translation t [tx, ty, tz], and speaking status ss [1/0] -
            Person A : [(r_A, t_A) * nposes, ss_A]
            Person B : [(r_B, r_B) * nposes, ss_B]

        The relative features for a single pose are computed as follows:
            relative rotation       - r_A * inverse(r_B)
            relative translation    - t_B - t_A
            relative speaking       - ss_B - ss_A

        Args:
            features    --  features tensor for people in the scene.
                            (batch_size(i.e. ngroups), npeople, features)

        Returns:
            The tensor of relative features of partners for each person
            (batch_size, npeople, npeople-1, features). Note that the rotation
            and location features are simply concatenated in the order
            [r * nposes, t * nposes, ss]

        """
        # Compute the tensor of partner features for every reference person
        partners = SocialPooler.compute_partners(features)
        # Expand reference features to match partners tensor size
        references = features.unsqueeze(2).repeat(1, 1, features.shape[1]-1, 1)
        # Get the indices for rotation and location features
        rot_idx = multi_range(4, self.nposes, 7)
        loc_idx = multi_range(3, self.nposes, 7, start=4)
        # Compute the relative features of partners wrt reference persons
        relative_rot = self.relative_rotation(
            partners[..., rot_idx].view(-1, 4),
            references[..., rot_idx].view(-1, 4)
        ).view(*partners.shape[:-1], self.nposes*4)
        relative_pos = partners[..., loc_idx] - references[..., loc_idx]
        relative_ss = partners[..., -1:] - references[..., -1:]
        return torch.cat((relative_rot, relative_pos, relative_ss), dim=-1)

    def forward(self, features: Tensor, embeddings: Tensor) -> Tensor:
        """ Compute the forward pass through the pooling module

        Args:
            features    --  the headposes and speaking status of the people in
                            a conversation, shape (batch_size, npeople, ninp).
                            last input dimension is expected to be the speaking
                            status if available.
            embeddings  --  the embedded feature representations,
                            shape (batch_size, npeople, nrep)

        Returns:
            the pooled representation for people in the group,
            shape (batch_size, npeople, nout)

        """
        # Compute the relative head poses, positions, and speaking status of
        # partners with respect to a reference person, for every person
        # shape (batch_size, npeople, npeople-1, feature_dim)
        relative_features = self.relative_features(features)
        # Embed the relative features
        # shape (batch_size, npeople, npeople-1, self.nhid)
        relative_embeddings = self.embedder(relative_features)
        # Rearrange the absolute feature embeddings to match the shape of
        # the tensor comprising features for each person
        # shape (batch_size, npeople, npeople-1, self.nrep)
        abs_embeddings = SocialPooler.compute_partners(embeddings)
        # Concatenate the relative embeddings with corresponding absolute
        # feature embeddings for the partners
        # shape (batch_size, npeople, npeople-1, self.nhid+self.nrep)
        partner_rep = torch.cat((relative_embeddings, abs_embeddings), dim=-1)
        # Process the partner representation for pooling
        # shape (batch_size, npeople, npeople-1, self.nout)
        pre_pooled = self.pre_pooler(partner_rep)
        # Max pool over the partners dimension to get final representation
        # shape (batch_size, npeople, self.nout)
        ngroups, npeople, npartners, f = pre_pooled.size()
        rep = pre_pooled.view(ngroups*npeople, npartners, f).transpose(1,2)
        out = F.adaptive_max_pool1d(rep, output_size=1).squeeze(-1)
        return out.view(ngroups, npeople, f)
