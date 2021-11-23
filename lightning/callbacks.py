#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: callbacks.py
# Created Date: Monday, October 5th 2020, 10:29:28 am
# Author: Chirag Raman
#
# Copyright (c) 2020 Chirag Raman
###


import itertools
import math
import pickle
import warnings
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from mathutils import Quaternion, Vector
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.metrics import Accuracy
from torch import Tensor
from torch.distributions import Normal

from common.tensor_op import multi_range
from data.types import (
    DataSplit, FeatureSet, Seq2SeqSamples, Seq2SeqPredictions, SerializationMap
)


def projected_difference(q1: Quaternion, q2: Quaternion,
                         track: Vector = Vector((0, -1, 0))) -> float:
    """ Applies q2 to track, projects onto xy, returns diff with q1 """
    track.rotate(q2)
    track.z = 0 # Project onto xy plane
    track.normalize()
    return math.degrees(track.to_track_quat("-Y", "Z")
                        .rotation_difference(q1).angle)


class TestSequenceProcessor(Callback):

    """ Base class that processes each test sequence prediction """

    def __init__(self, out_dir: Path, train_mean: Tensor = torch.tensor(0.0),
                 train_std: Tensor = torch.tensor(1.0)):
        """ Initialize the callback object """
        super().__init__()
        self.__init_output_dir(out_dir)
        self.train_mean = train_mean
        self.train_std = train_std

    def __init_output_dir(self, out_dir: Path):
        """ Check if directory already exists, and create one """
        if out_dir.is_file():
            raise ValueError("Serializer expected directory path, got filepath"
                             " instead: {}".format(str(out_dir)))
        if out_dir.is_dir():
            warnings.warn("Serialization output directory already exists, "
                          "its contents will be overwritten!")
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir

    def split_test_sequences(
            self, outputs: Seq2SeqPredictions, target: Seq2SeqSamples
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """ Yield each indivudual test input sequence and prediction
        Takes a mean over all z samples in the prediction, and assumed ground
        truth future to be present.
        Input samples consist of observed tensors of shape
        (seq_len, batch_size, npeople, data_dim)
        Predicted Normal loc and scale are of shape
        (nz_samples, seq_len, batch_size, npeople, data_dim)
        Returns a tuple of tensors corresponding to
        (observed, observed_start, offset, future_gt, future_mean, future_std)
        """
        # HACK: Take mean over nz_samples, assuming nz_samples=1
        future_means = outputs.stochastic.loc.mean(0)
        future_scales = outputs.stochastic.scale.mean(0)
        seqs = zip(
            target.observed.split(1, dim=1),
            target.observed_start,
            target.offset,
            target.future.split(1, dim=1),
            future_means.split(1, dim=1),
            future_scales.split(1, dim=1)
        )
        return seqs

    def detach_tensors(
        self, outputs: Seq2SeqPredictions, batch: DataSplit
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """ Return the target observed and future, and predicted means and std """
        # Input samples consist of tensors of shape
        # (seq_len, nseq, npeople, data_dim)
        obs = batch.target.observed.contiguous().detach().cpu()
        fut = batch.target.future.contiguous().detach().cpu()
        # Predicted Normal loc and scale are of shape
        # (nz_samples, seq_len, nseq, npeople, data_dim)
        fut_means = outputs.stochastic.loc.contiguous().detach().cpu()
        fut_scales = outputs.stochastic.scale.contiguous().detach().cpu()
        return obs, fut, fut_means, fut_scales

    def denormalize(
            self, outputs: Seq2SeqPredictions, batch: DataSplit
    ) -> Tuple[Seq2SeqSamples, Seq2SeqPredictions]:
        """ Destandardize the stochastic predictions and inputs """
        obs, fut, fut_means, fut_scales = self.detach_tensors(outputs, batch)

        # Destandardize inputs and predictions before computing metric
        normed = [obs, fut, fut_means]
        denormed = [(t * self.train_std) + self.train_mean for t in normed]
        # Scale the std by the same scaling factor as the means
        fut_scales *= self.train_std

        return *denormed, fut_scales


class TestSerializer(TestSequenceProcessor):

    """ Serialize test outputs from the SocialProcess model.
    Results are pickled as well as writted to csv for visualization.
    """

    def __init__(
            self, out_dir: Path, time_stride: int,
            train_mean: Tensor = torch.tensor(0.0),
            train_std: Tensor = torch.tensor(1.0),
            serialize_batch: bool = False, serialize_seq: bool = True,
            group_map: Optional[SerializationMap] = None
        ) -> None:
        """ Initialize the serializer object
        Args:
            out_dir         --  directory to store outputs
            time_stride     --  the sampling rate of the sequences
            serialize_batch --  if False, do not serialize the entire batch
            serialize_seq   --  if False, do not serialize individual sequences
            map             --  The SerializationMap specifying the group_ids
                                and corresponding observed start ranges to
                                serialize. Ignored if serialize_seq is False.
        """
        super().__init__(out_dir, train_mean, train_std)
        self.time_stride = time_stride
        self.serialize_seq = serialize_seq
        self.serialize_batch = serialize_batch
        self.group_map = group_map

        if not serialize_batch and not serialize_seq:
            warnings.warn("Test Serializer is configured to not serialize "
                          "anything. Do you need one?")

    def on_test_batch_end(
            self, trainer: Trainer, pl_module: LightningModule,
            outputs: Seq2SeqPredictions, batch: DataSplit, batch_idx: int,
            dataloader_idx: int
        ) -> None:
        """ Serialize test outputs to self.out_dir
        Pickle the list of dictionaries. Write each sequence and future pair
        to a separate pickle file.
        """
        # Denormalize stochastic predictions and inputs
        # TODO: Refactor types to dataclass instead of named tuple
        obs, fut, fut_means, fut_scales = self.denormalize(outputs, batch)
        target = batch.target
        target = Seq2SeqSamples(
            key=target.key, observed_start=target.observed_start,
            observed=obs, future_len=target.future_len,
            offset=target.offset, future=fut
        )
        predictions = Seq2SeqPredictions(
            stochastic=Normal(fut_means, fut_scales),
            posteriors=outputs.posteriors,
            deterministic=outputs.deterministic
        )


        # Pickle the inputs and outputs
        if self.serialize_batch:
            filename = "all-outputs-{}_{}.pkl".format(dataloader_idx, batch_idx)
            with open(self.out_dir/filename, "wb") as fh:
                pickle.dump({"inputs": target, "predictions": predictions}, fh)

        if self.serialize_seq:
            # Check if the batch is from a group to serialize
            bounds = -1
            if self.group_map:
                if batch.target.key not in self.group_map.keys():
                    return
                else:
                    bounds = self.group_map[batch.target.key]
            # Pickle each observed sequence and prediction to a separate file
            enum_seqs = enumerate(self.split_test_sequences(predictions, target))
            for _, (obs, start, offset, fut, fut_mean, fut_std) in enum_seqs:
                # Skip serializing if start is not in self.obs_start_bound
                if (bounds != -1
                        and not any([start in range(b[0], b[1])
                                     for b in bounds])):
                    continue
                # Create the file as
                # "group_id=<group_id>-start=<start>-offset=<offset>.pkl"
                filename = "group_id={}-start={}-offset={}.pkl".format(
                    target.key, int(start), int(offset)
                )
                payload = {
                    "observed": obs.squeeze(1).numpy(),
                    "observed_start": int(start),
                    "offset": int(offset),
                    "future": fut.squeeze(1).numpy(),
                    "fut_mean": fut_mean.squeeze(1).numpy(),
                    "fut_scale": fut_std.squeeze(1).numpy(),
                    "time_stride": self.time_stride
                }
                with open(self.out_dir / filename, "wb") as fh:
                    pickle.dump(payload, fh)


class MetricsComputer(TestSequenceProcessor):

    """ Compute evaluation metrics.
    Each sequence has a series of summary metrics and timestep metrics.
    These include nll, body and head location and rotation errors, and
    speaking status accuracy. The dataframe of metrics is stored in the
    specified output directory with the name `test_metrics.h5`.
    """

    def __init__(self, out_dir: Path, nposes: int, ntimesteps: int,
                 time_stride: int, train_mean: Tensor, train_std: Tensor,
                 feature_set: FeatureSet = FeatureSet.HBPS, project_rotation: bool = True) -> None:
        """ Initialize the callback object """
        super().__init__(out_dir, train_mean, train_std)
        self.nposes = nposes
        self.nsteps = ntimesteps
        self.stride = time_stride
        self.feature_set = feature_set
        self.project_rotation = project_rotation
        self.batch_metrics = []

    @staticmethod
    def info_columns() -> List[str]:
        """ Column names for the sequence information in the output dataframe """
        return ["group_id", "obs_start", "offset"]

    @staticmethod
    def _root_metric_names(nposes: int, project_rot: bool = True) -> List[str]:
        """ Prefix and suffix free names of metrics to use as columns
        Internal method; "mean_" or "_ts" is added to these by other static
        methods in this class. Returned list does not contain speaking metrics.
        """
        poses = ["body", "head"] if nposes == 2 else ["head"]
        means_metrics = ["loc_err", "rot_err"]
        metrics = [p+"_"+m  for m in means_metrics for p in poses]
        if project_rot:
            proj_metrics = [p+"_projected_rot_err" for p in poses]
            metrics.extend(proj_metrics)
        metrics = ["nll", *metrics]
        return metrics

    @staticmethod
    def summary_columns(feature_set: FeatureSet, nposes: int,  project_rot: bool = True) -> List[str]:
        """ Column names for the seq summary metrics in the output dataframe """
        cols =  ["mean_"+m for m in MetricsComputer._root_metric_names(nposes, project_rot)]
        if feature_set == FeatureSet.HBPS:
            cols.append("speaking_accuracy")
        return cols

    @staticmethod
    def ts_columns(feature_set: FeatureSet, nposes: int,  project_rot: bool = True) -> List[str]:
        """ Column names for the timestep metrics in the output dataframe """
        cols = [m+"_ts" for m in MetricsComputer._root_metric_names(nposes, project_rot)]
        if feature_set == FeatureSet.HBPS:
            cols.append("speaking_accuracy_ts")
        return cols

    def df_columns(self) -> pd.MultiIndex:
        """ Computes the multi-index for the metrics dataframe """
        # Create top level columns for summary and ts metrics
        info_cols = MetricsComputer.info_columns()
        summary_cols = MetricsComputer.summary_columns(self.feature_set, self.nposes, self.project_rotation)
        ts_cols = MetricsComputer.ts_columns(self.feature_set, self.nposes, self.project_rotation)
        ts_cols_repeated = list(itertools.chain.from_iterable(
            itertools.repeat(m, self.nsteps) for m in ts_cols
        ))
        cols = [*info_cols, *summary_cols, *ts_cols_repeated]
        # Second level columns for timesteps
        timesteps = [*range(0, self.nsteps*self.stride, self.stride)]
        s_cols = [*[""]*(len(info_cols) + len(summary_cols)),
                  *timesteps*len(ts_cols)]
        # Create the pd.MultiIndex and return
        return pd.MultiIndex.from_arrays((cols, s_cols))

    def nll_timestep(
            self, future_means: Tensor, future_scales: Tensor, future: Tensor
        ) -> np.ndarray:
        """ Compute nll per timestep
        Inputs of shape (seq_len, nsequences, npeople, data_dim)
        Returns mean nll per timestep:
            mean_nll            -- np.ndarray (nsequences, seq_len)
        """
        nll = - Normal(future_means, future_scales).log_prob(future)
        nll = nll.sum(dim=(2, 3))
        return nll.transpose(0, 1).contiguous().numpy()

    def location_errors_timestep(
            self, future_means: Tensor, future: Tensor
        ) -> List[np.ndarray]:
        """ Compute mean mse in location per timestep and sequence
        Returns a list of the mse per timestep for each pose (body, head, etc)
        """
        loc_idx = multi_range(3, self.nposes, 7, start=4)
        dist = []
        for i in range(self.nposes):
            # Compute one pose error at a time
            idxs = loc_idx[i*3:(i+1)*3]
            # (nseq, seq_len, npeople, 1)
            gt = future[..., idxs]
            pred = future_means[..., idxs]
            b, t, p, f = gt.size()
            # (nseq * seq_len * npeople)
            p2_dist = F.pairwise_distance(gt.view(-1, 3), pred.view(-1, 3))
            # (nseq, seq_len)
            p2_dist_timestep = p2_dist.view(b, t, p).mean(-1)
            # Store the results
            dist.append(p2_dist_timestep.numpy())
        return dist

    def rotation_errors_timestep(
            self, future_means: Tensor, future: Tensor
        ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """ Compute error in orientation per timestep and sequence
        Returns two lists for the absolute, and reprojected, error in degrees
        per timestep for every pose. The second list is empty if
        self.project_rotation is False
        """
        nseqs, seq_len, npeople, _ = future_means.size()
        rot_idx = multi_range(4, self.nposes, 7)
        rot_err = []
        proj_rot_err = []
        for i in range(self.nposes):
            # Compute one pose error at a time
            idxs = rot_idx[i*4:(i+1)*4]
            preds = future_means[..., idxs].view(-1, 4)
            pred_quats = [Quaternion(v) for v in preds]
            gts = future[..., idxs].view(-1, 4)
            gt_quats = [Quaternion(v) for v in gts]
            # Absolute rotation difference
            err = [math.degrees(qg.rotation_difference(qp).angle)
                   for qg, qp in zip(gt_quats, pred_quats)]
            err = np.array(err).reshape((nseqs, seq_len, npeople)).mean(-1)

            # Store the errors
            rot_err.append(err) # (nseqs, seq_len)

            if self.project_rotation:
                # Project tracking vector onto xy plane and compute error,
                # effectively measuring rotation error around z axis
                proj_err = [projected_difference(qg, qp)
                            for qg, qp in zip(gt_quats, pred_quats)]
                proj_err = np.array(proj_err).reshape((nseqs, seq_len, npeople))
                proj_err = proj_err.mean(-1)
                proj_rot_err.append(proj_err) # (nseqs, seq_len)

        return rot_err, proj_rot_err

    def speaking_errors(
            self, future_means: Tensor, future: Tensor
        ) -> Tuple[np.ndarray, np.ndarray]:
        """ Compute the accuracy in speaking status per timestep """
        # (nseq, seq_len, npeople)
        pred = future_means[..., -1]
        gt = future[..., -1]
        accuracy = Accuracy()
        nseq, seq_len, npeople = pred.size()

        # accuracy per sequence
        acc = [accuracy(p, g).item() for p, g in zip(pred, gt)]
        acc = np.array(acc)[:, np.newaxis]

        # accuracy per timestep
        ts_acc = [accuracy(p, g).item()
                  for p, g in zip(pred.view(-1, npeople), gt.view(-1, npeople))]
        ts_acc = np.array(ts_acc).reshape(nseq, seq_len)

        return acc, ts_acc

    def on_test_batch_end(
            self, trainer: Trainer, pl_module: LightningModule,
            outputs: Seq2SeqPredictions, batch: DataSplit, batch_idx: int,
            dataloader_idx: int
        ) -> None:
        """ Compute test metrics for each sequence in the batch
        Metrics are stored in a pandas DataFrame in self.out_dir at the end of
        the test epoch.
        """
        # 1. Compute Negative Log Likelihood - mean, per timestep on non denormalized data
        nll_ts = self.nll_timestep(
            outputs.stochastic.loc.mean(0).detach().cpu(),
            outputs.stochastic.scale.mean(0).detach().cpu(),
            batch.target.future
        ) # (nseq, seq_len)
        mean_nll = nll_ts.mean(1, keepdims=True) # (nseq, 1)

        # Denormalize stochastic predictions and inputs
        _, fut, fut_means, fut_scales = self.denormalize(outputs, batch)

        # HACK: Take mean over nz_samples and transpose to get
        # (nseq, seq_len, npeople, data_dim) (assumes 1 nz_samples)
        fut_means = fut_means.mean(0)
        fut_scales = fut_scales.mean(0)

        # Samples consist of tensors of shape (seq_len, nseq, npeople, data_dim)
        # Convert to (nseq, seq_len, ...)
        fut, fut_means, fut_scales = [t.transpose(0, 1).contiguous()
                                      for t in [fut, fut_means, fut_scales]]

        # Compute the rest of the metrics
        # 2. MSE for body loc and head loc - mean, per timestep
        dist_ts = self.location_errors_timestep(fut_means, fut) # (nseq, seq_len)
        mean_dist = [err.mean(1, keepdims=True) for err in dist_ts] # (nseq, 1)

        # 3. Error in degrees for body rot and head rot - mean, per timestep
        rot_err_ts, proj_rot_err_ts = \
            self.rotation_errors_timestep(fut_means, fut) # (nseq, seq_len)
        mean_rot_err = [err.mean(1, keepdims=True) for err in rot_err_ts] # (nseq)

        # Store in lists to stack
        mean_metrics = [mean_nll, *mean_dist, *mean_rot_err]
        ts_metrics = [nll_ts, *dist_ts, *rot_err_ts]

        # Add projected metrics if needed
        if self.project_rotation:
            mean_proj_rot_err = [err.mean(1, keepdims=True) for err in proj_rot_err_ts]
            mean_metrics.extend(mean_proj_rot_err)
            ts_metrics.extend(proj_rot_err_ts)

        # 4. Speaking status error
        # (nseq, 1), (nseq, seq_len)
        if self.feature_set == FeatureSet.HBPS:
            speaking_acc, speaking_acc_ts = self.speaking_errors(fut_means, fut)
            # Add speaking status at the end
            mean_metrics.append(speaking_acc)
            ts_metrics.append(speaking_acc_ts)

        # Include group_id, obs_start, offset
        g_id = np.array([batch.target.key]*fut.size(0))[:, np.newaxis]
        obs_start = (batch.target.observed_start
                     .detach().cpu().numpy()[:, np.newaxis])
        offset = batch.target.offset.detach().cpu().numpy()[:, np.newaxis]

        # Stack horizontally to form dataframe for the batch
        metrics = np.hstack((g_id, obs_start, offset, *mean_metrics, *ts_metrics))

        # Append to dataframes list
        self.batch_metrics.append(metrics)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """ Stack individual batch metrics, create dataframe, and serialize """
        # Create and save the metrics df
        metrics = np.vstack(self.batch_metrics)
        df = pd.DataFrame(data=metrics, columns=self.df_columns())
        df.to_hdf(self.out_dir/"test_metrics.h5", key="test_metrics")
        # Compute statistics over the summary columns and write to csv
        summary_cols = MetricsComputer.summary_columns(self.feature_set, self.nposes, self.project_rotation)
        summary_metrics = df[summary_cols]
        summary_metrics = summary_metrics.apply(pd.to_numeric)
        stats = summary_metrics.describe()
        stats[~stats.index.isin(["25%", "50%", "75%"])].to_csv(
            self.out_dir/"summary_metrics.csv", index_label="statistic"
        )
