#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: loss.py
# Created Date: Sunday, December 1st 2019, 6:31:42 pm
# Author: Chirag Raman
#
# Copyright (c) 2019 Chirag Raman
###


from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from common.tensor_op import multi_range
from data.types import DataSplit, FeatureSet, Seq2SeqPredictions
from .elbo import SocialProcessSeq2SeqElbo


class SocialProcessLoss(nn.Module):

    """ Compute the loss for training a SocialProcess """

    def __init__(self, aux_criterion: nn.Module = None):
        """ Initialize the module.
        Args:
            aux_criterion   :   A module that computes any auxillary loss
                                terms barring regularization
        """
        super().__init__()
        self.elbo = SocialProcessSeq2SeqElbo()
        self.aux_criterion = aux_criterion

    def forward(
            self, sp_prediction: Seq2SeqPredictions, split: DataSplit
        ) -> Tuple[Tensor, Tensor]:
        """ Compute the loss """
        trg = split.target
        loss, nll, kl = self.elbo(sp_prediction, trg.future)
        aux_losses = None
        if self.aux_criterion is not None:
            future_mean = sp_prediction.stochastic.loc
            aux_losses = self.aux_criterion(
                future_mean, trg.future.expand_as(future_mean)
            )
            for pred in sp_prediction.deterministic:
                # Deterministic decoded futures from the latent and det paths
                if pred is not None:
                    aux_losses += self.aux_criterion(pred, trg.future)
            loss += aux_losses
            aux_losses = aux_losses.detach().clone()
        return loss, nll, kl, aux_losses


class GeometricHomoscedasticLoss(nn.Module):

    """ Learn weighting to balance positional and rotational loss

    Based on "Geometric loss functions for camera pose regression with
    deep learning"

    Args:
        sx      --  Homoscedastic uncertainty in position
        sq      --  Homoscedastic uncertainty in orientation
        nposes  --  The number of poses for a single person (for eg. nposes = 2
                    if both head and body pose features are being predicted)
        nrot    --  Number of dimensions of rotation
                    (4 for quaternion, 3 for Euler)

    """

    def __init__(self, sx: float = 0, sq: float = -3, nposes: int = 1) -> None:
        """ Initialise the object """
        super().__init__()
        self.sx = nn.Parameter(torch.tensor(sx))
        self.sq = nn.Parameter(torch.tensor(sq))
        self.nposes = nposes
        self.criterion = nn.MSELoss()

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """ Compute the loss by weighting positional and rotational terms

        Args:
            prediction  --  The predicted features expected to be of shape
                            (..., <(qw, qx, qy, qz, tx, ty, tz) * nposes, ...>)
            target      --  The ground truth features expected to be of shape
                            (..., <(qw, qx, qy, qz, tx, ty, tz) * nposes, ...>)
        """
        # Get the indices for rotation and location features
        rot_idx = multi_range(4, self.nposes, 7)
        loc_idx = multi_range(3, self.nposes, 7, start=4)
        # Compute the weighted loss
        loss_rot = self.criterion(prediction[..., rot_idx], target[..., rot_idx])
        loss_pos = self.criterion(prediction[..., loc_idx], target[..., loc_idx])
        weighted_loss = ((-1 * self.sx).exp() * loss_pos + self.sx
                         + (-1 * self.sq).exp() * loss_rot + self.sq)
        return weighted_loss


class SocialAuxLoss(nn.Module):

    """ Combine the geometric and speaking status losses

    The homoscedastic loss is used to regress orientation and location while
    binary cross-entropy with logits is computed for speaking status

    """

    def __init__(self, sx: float = 0, sq: float = -3, nposes: int = 1,
                 feature_set : FeatureSet = FeatureSet.HBPS) -> None:
        """ Initialize the loss module """
        super().__init__()
        self.geometric = GeometricHomoscedasticLoss(sx, sq, nposes)
        self.feature_set = feature_set

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """ Compute the loss by weighting positional and rotational terms

        Args:
            prediction  --  The predicted features expected to be of shape
                            (..., <(qw, qx, qy, qz, tx, ty, tz) * nposes, ss (optional)>)
            target      --  The ground truth features expected to be of shape
                            (..., <(qw, qx, qy, qz, tx, ty, tz) * nposes, ss (optional)>)
        """
        # Compute Geometric Loss
        loss = self.geometric(prediction, target)
        if self.feature_set == FeatureSet.HBPS:
            # Compute speaking status loss
            loss += F.binary_cross_entropy_with_logits(
                prediction[..., -1], target[..., -1]
            )
        return loss