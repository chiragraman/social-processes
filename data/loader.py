#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: loader.py
# Created Date: Saturday, November 16th 2019, 9:42:20 pm
# Author: Chirag Raman
#
# Copyright (c) 2019 Chirag Raman
###


import copy
import itertools
import logging
from random import shuffle
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Sampler
from torch.utils.data.dataloader import default_collate

from .datasets import BucketMap, SocialDatasetInterface
from .types import DataSplit, Seq2SeqSamples


COLLATE_TYPE = Union[
    Callable[[Sequence[Seq2SeqSamples], float], DataSplit],
    Callable[[Sequence[Seq2SeqSamples], int], DataSplit],
    Callable[[Sequence[Tuple[Seq2SeqSamples, np.ndarray]]], DataSplit],
    Callable[[Sequence[Tuple[Seq2SeqSamples, Sequence[Seq2SeqSamples]]]],
             DataSplit]
]


def collate_seq2seq(batch: Sequence[Seq2SeqSamples]) -> Seq2SeqSamples:
    """ Repack a default collated batch of samples

    Transpose the data tensors to be sequence first, and replace key with the
    group_id

    """
    g_id = batch[0].key[0] # key is a tuple of (group id, group size)
    obs_start = default_collate([s.observed_start for s in batch])
    obs = default_collate([s.observed for s in batch])
    obs = obs.transpose(1, 0).contiguous()
    future_len = batch[0].future_len # batches are bucketed by seq_len
    offset = default_collate([s.offset for s in batch])
    # If futures are None, set batch future to None. Note that having
    # only some futures as None in a batch will lead to an error. To get a
    # minor speedup, only checking if the first sample future is None instead
    # of checking for all samples.
    fut = [s.future for s in batch]
    fut = None if fut[0] is None else (default_collate(fut)
                                       .transpose(1, 0).contiguous())
    return Seq2SeqSamples(
        key=g_id, observed_start=obs_start, observed=obs,
        future_len=future_len, offset=offset, future=fut
    )

def collate_random_context(
        batch: Sequence[Seq2SeqSamples], max_context: float = 0.8
    ) -> DataSplit:
    """ Split the batch into context and target data descriptors, Following the
    conventions in "Empirical Evaluation of Neural Process Objectives",
    the context points are chosen as a subset of the target points.

    Args:
        batch       : The batch from the data loader to collate
        max_context : The maximum fraction of the batch size that can be
                      used as context samples

    """
    ncontext = torch.randint(3, int(max_context*len(batch)), ())
    return collate_sampled_context(batch, ncontext=ncontext)


def collate_sampled_context(
        batch: Sequence[Seq2SeqSamples], ncontext: int = 32
    ) -> DataSplit:
    """ Split the batch into context and target data descriptors, Following the
    conventions in "Empirical Evaluation of Neural Process Objectives",
    the context points are chosen as a subset of the target points.

    Args:
        batch       : The batch from the data loader to collate
        ncontext    : The number of context points for every batch; the first
                      `ncontext` points are chosen as context

    """
    context = collate_seq2seq(batch[:ncontext])
    target = collate_seq2seq(batch)
    return DataSplit(context=context, target=target)


def collate_unpaired_context(
        batch: Sequence[Tuple[Seq2SeqSamples, np.ndarray]]
    ) -> DataSplit:
    """ Collate a batch of samples for evaluation

    The batch is expected to already contain the context data common
    for the batch

    """
    # Since a single batch contains sequences from a single group, assume
    # the same context for the entire batch. Each batch item is a tuple of
    # (target samples, ctx data)
    g_id = batch[0][0].key[0] # key is a tuple of (group id, group size)
    ctx_data = batch[0][1]
    ctx_data = torch.as_tensor(ctx_data).contiguous()
    # Wrap context tensor in a dummy Seq2Seq structure for api uniformity
    context = Seq2SeqSamples(key=g_id, observed_start=0, observed=ctx_data,
                             future_len=0, offset=0, future=None)

    # Collate the target Seq2Seq pairs
    target = [s[0] for s in batch]
    target = collate_seq2seq(target)
    return DataSplit(context=context, target=target)


def collate_paired_context(
        batch: Sequence[Tuple[Seq2SeqSamples, Sequence[Seq2SeqSamples]]]
    ) -> DataSplit:
    """ Collate a batch of samples with observed, future pairs for context

    The batch is expected to have a list of observed, future pairs common
    to the entire batch

    """
    # Since a single batch contains sequences from a single group, assume
    # the same context for the entire batch. Each batch item is a tuple of
    # (target samples, List[context samples])
    context = batch[0][1] # list of context samples for the entire batch
    # Collate context pairs
    context = collate_seq2seq(context)
    # Collate the target pairs
    target = [s[0] for s in batch]
    target = collate_seq2seq(target)
    return DataSplit(context=context, target=target)


def split_group(batch: Sequence[Seq2SeqSamples]) -> DataSplit:
    """ Split a batch of conversing groups into context and target descriptors.

    The context points are sampled along the people dimension, in the range
    [2, npeople). Following the conventions in "Empirical Evaluation of Neural
    Process Objectives", the context points are chosen as a subset of the
    target points.

    """
    g_id = batch[0].key[0]
    target = collate_seq2seq(batch)
    # Each data tensor is of shape
    # (seq_len, batch_size, npeople, data_dim)
    _, _, npeople, _ = target.observed.size()
    ncontext = torch.randint(2, npeople, ()).item()
    ctx_indices = np.random.choice(range(npeople),
                                   size=ncontext, replace=False)
    context = Seq2SeqSamples(
        key=g_id,
        observed_start=target.observed_start,
        observed=target.observed[:, :, ctx_indices, :],
        future_len=target.future_len,
        offset=target.offset,
        future=target.future[:, :, ctx_indices, :]
    )
    return DataSplit(context=context, target=target)


class GroupSampler(Sampler):

    """ Sample batches of same seq_len and group size """

    def __init__(
            self, data_source: SocialDatasetInterface, batch_size: int,
            min_batch_size: int = 5, fixed_batches: Optional[Sequence] = None
        ) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.fixed_batches = fixed_batches
        if fixed_batches is None:
            # Make an initial call to compute batches to compute
            # number of samples
            self.compute_batches()
        else:
            self.num_samples = len(fixed_batches)

    def compute_batches(self) -> List:
        """ Compute the list of batches """
        batches = []
        for key, obs_map in self.data_source.bucket_map.items():
            # Shuffle order of observed sequences
            obs_seqs = copy.deepcopy(list(obs_map.values()))
            shuffle(obs_seqs)
            # Shuffle indices within idx list for each observed idx
            for idx_list in obs_seqs:
                shuffle(idx_list)
            # Transpose indices to ensure unique obs seqs in each batch; eg.
            # [[1,2,3], [4,5,6,7], [8,9]] ->
            # [[1,4,8], [2,5,9], [3,6,None], [None,7,None]]
            indices = list(map(list, itertools.zip_longest(*obs_seqs)))
            # Flatten transposed indices; eg.
            # [[1,4,8], [2,5,9], [3,6,None], [None,7,None]] ->
            # [1, 4, 8, 2, 5, 9, 3, 6, 7]
            indices = [idx for idx_list in indices for idx in idx_list
                       if idx is not None]
            # If number of samples for current key is less than minimum
            # batch size, skip the current key
            if len(indices) < self.min_batch_size:
                logging.info(f"[!] Not enough seqs for key {key} for batching")
                continue
            # Create batches from flattened indices
            for seqs in [indices[i:i+self.batch_size]
                         for i in range(0, len(indices), self.batch_size)]:
                if len(seqs) < self.min_batch_size:
                    # If the batch is smaller than allowed minimum, extend
                    # the last batch with the current batch of sequences
                    batches[-1].extend(seqs)
                else:
                    # Create a new batch with the sequences
                    batches.append(seqs)
        self.num_samples = len(batches)
        return batches

    def __iter__(self):
        if self.fixed_batches is None:
            batches = self.compute_batches()
            # Shuffle batches so that they aren't ordered by bucket size
            shuffle(batches)
        else:
            batches = self.fixed_batches
        for batch in batches:
            yield batch

    def __len__(self):
        return self.num_samples
