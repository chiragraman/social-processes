#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: preprocessing.py
# Created Date: Thursday, January 1st 1970, 1:00:00 am
# Author: Chirag Raman
#
# Copyright (c) 2020 Chirag Raman
###


import abc
from math import radians
from typing import Sequence
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
from mathutils import Vector, Euler

from common.graphics_utils import convert_loc_rh_y_down_to_z_up, convert_normals_to_quaternions


FRAME_COL = "frame"
GID_COL = "group_id"
GSIZE_COL = "group_size"
PID_COL = "person_id"
INFO_COLUMNS = [FRAME_COL, GID_COL, GSIZE_COL, PID_COL]
BASIC_FEATURES_COLUMNS = [
    "body_qw", "body_qx", "body_qy", "body_qz", "body_tx",
    "body_ty", "body_tz", "head_qw", "head_qx", "head_qy",
    "head_qz", "head_tx", "head_ty", "head_tz", "speaking"
]


def assemble_group_df(
        group_id: str, group_size: int, seq_len: int, frames: np.ndarray,
        person_ids: Sequence, interleaved_features: np.ndarray, columns: Sequence
    ) -> pd.DataFrame:
    """ Compute the final features dataframe from a conversation group """
    frames = np.repeat(frames, group_size)
    gids = [group_id] * seq_len * group_size
    gsizes = [group_size] * seq_len * group_size
    pids = np.tile(person_ids, seq_len)
    # Construct the group dataframe
    info_df = pd.DataFrame(
        {
            FRAME_COL: frames,
            GID_COL: gids,
            GSIZE_COL: gsizes,
            PID_COL: pids
        }
    )
    data_df = pd.DataFrame(interleaved_features.astype(np.float32),
                           columns=columns)

    return pd.concat([info_df, data_df], axis=1)


class FeatureExtractorInterface(metaclass=abc.ABCMeta):

    """ Abstract Base Class for feature extractors """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, "feature_fields")
                and callable(subclass.feature_fields)
                and hasattr(subclass, "extract")
                and callable(subclass.extract))

    @classmethod
    @abc.abstractmethod
    def feature_fields(cls) -> List[str]:
        """  Labels for the feature fields """
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, *args, **kwargs) -> np.ndarray:
        """ Extract features from some raw data """
        raise NotImplementedError


class PanopticBasicFeatures(FeatureExtractorInterface):
    """ Compute pose and speaking status features for a person in the Haggling Dataset """

    @classmethod
    def feature_fields(cls) -> List[str]:
        """ Names for the feature columns """
        return BASIC_FEATURES_COLUMNS

    def extract(self, body: Dict, speech: Dict) -> np.ndarray:
        """ Compute feature array for a subject """
        # Assert that all frames are valid
        assert body["bValid"]
        # body pose shape (nframes, 7), consisting of (quaternion, location)
        body_normal = convert_loc_rh_y_down_to_z_up(body["bodyNormal"].transpose())
        body_quat = convert_normals_to_quaternions(body_normal)
        neck_loc = body["joints19"][0:3, :].transpose() # (nframes, 3)
        neck_loc = convert_loc_rh_y_down_to_z_up(neck_loc) # (nframes, 3)
        # head pose shape (nframes, 7)
        head_normal = convert_loc_rh_y_down_to_z_up(body["faceNormal"].transpose())
        head_quat = convert_normals_to_quaternions(head_normal)
        nose_loc = body["joints19"][3:6, :].transpose() # (nframes, 3)
        nose_loc = convert_loc_rh_y_down_to_z_up(nose_loc)
        # speaking status shape (nframes, 1)
        speaking = speech["indicator"][:, None]
        # Trim features to have the same shape
        nframes = min([arr.shape[0] for arr in
                       (body_quat, neck_loc, head_quat, nose_loc, speaking)])
        # Concatenate features
        return np.hstack(
            (body_quat[:nframes], neck_loc[:nframes], head_quat[:nframes],
             nose_loc[:nframes], speaking[:nframes])
        )


def convert_panoptic_group(
        group_id: str, data: List[Dict], extractor: FeatureExtractorInterface
    ) -> pd.DataFrame:
    """ Convert group data into Dataframes expected by SocialDataset """
    # Assert order of humans in all dictionaries, and the startFrame, match
    body, speech = data
    assert (
        [sub["humanId"] for sub in body["subjects"]] \
        == [sub["humanId"] for sub in speech["speechData"]]
    ), ("Order of participants should match in the data dictionaries")
    assert (body["startFrame"] == speech["startFrame"]), (
        "body data and speech data start frames don't match"
    )

    # Get features for each subject. Expected (nframes, data_dim)
    gsize = len(body["subjects"])
    features = [
        extractor.extract(b, s) for (b, s) in zip(
            body["subjects"], speech["speechData"]
        )
    ]
    seq_len, data_dim = features[0].shape
    interleaved = np.hstack(features).reshape(seq_len * gsize, data_dim)

    frames = np.arange(body["startFrame"], body["startFrame"] + seq_len)
    person_ids = [sub["humanId"] for sub in body["subjects"]]
    group_df = assemble_group_df(
        group_id, gsize, seq_len, frames, person_ids,
        interleaved, extractor.feature_fields()
    )
    return group_df

class MNMBasicFeatures(FeatureExtractorInterface):
    """ Compute pose features for a person in the MnM Dataset"""

    BODY_Z = 25
    HEAD_Z = 40

    @classmethod
    def feature_fields(cls) -> List[str]:
        """ Names for the feature columns """
        return BASIC_FEATURES_COLUMNS[:-1]

    @classmethod
    def _normal_for_angle(cls, degree: float) -> Vector:
        """ Rotate the positive x direction by angle around +Z """
        x_dir = Vector((1, 0, 0))
        x_dir.rotate(Euler((0, 0, radians(degree)), "XYZ"))
        return x_dir

    def extract(self, raw_features: pd.DataFrame) -> np.ndarray:
        """ Compute feature array for a subject """
        # Remove the outer p_id level to make access easier
        raw_features.columns = raw_features.columns.droplevel(0)
        # body pose shape (nframes, 7), consisting of (quaternion, location)
        body_normal = np.array([self._normal_for_angle(degree)
                                for degree in raw_features["bOrient_deg"]])
        body_quat = convert_normals_to_quaternions(body_normal)
        body_x = raw_features[["sh1X", "sh2X"]].mean(axis=1).values[:, np.newaxis] # (nframes,)
        body_y = raw_features[["sh1Y", "sh2Y"]].mean(axis=1).values[:, np.newaxis] # (nframes,)
        body_z = np.repeat(self.BODY_Z, len(body_x))[:, np.newaxis]
        # head pose shape (nframes, 7)
        head_normal = np.array([self._normal_for_angle(degree)
                                for degree in raw_features["hOrient_deg"]])
        head_quat = convert_normals_to_quaternions(head_normal)
        head_xy = raw_features[["headX", "headY"]].values
        head_z = np.repeat(self.HEAD_Z, len(head_xy))[:, np.newaxis]
        # Concatenate features
        return np.hstack(
            (body_quat, body_x, body_y, body_z, head_quat, head_xy, head_z)
        )


def convert_mnm_group(
        group_id: str, group_size: int, data: pd.DataFrame,
        extractor: FeatureExtractorInterface
    ) -> pd.DataFrame:
    """ Convert the raw features into dataframes expected by a SocialDataset

    Args:
        group_id    -- The string representing the group id
        group_size  -- The number of participants in the group
        data        -- Raw features from the MnM dataset. Expects a multiindex
                       column where the outer level denotes person_id and the
                       inner level denotes raw features. Expects the following
                       feature colums :
                       ['frameId', 'visible', 'cameraNum', 'sh1X', 'sh1Y', 'headX',
                        'headY', 'sh2X', 'sh2Y', 'gazeX', 'gazeY', 'hOrient_deg',
                        'bOrient_deg']
        extractor   -- feature extractor for a single person

    """
    # Assert that no NaN values exist
    assert not data.isnull().values.any()
    # Iterate over participants
    features = [extractor.extract(p_raw) for _, p_raw in data.groupby(level=0, axis=1)]
    seq_len, data_dim = features[0].shape
    interleaved = np.hstack(features).reshape(seq_len * group_size, data_dim)

    pids = data.columns.levels[0]
    group_df = assemble_group_df(
        group_id, group_size, seq_len, data[(pids[0], "frameId")].values, pids,
        interleaved, extractor.feature_fields()
    )
    return group_df