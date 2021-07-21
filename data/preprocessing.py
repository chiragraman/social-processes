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
from typing import Callable
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
from mathutils import Quaternion
from mathutils import Vector


INFO_COLUMNS = ["frame", "group_id", "group_size", "person_id"]


def convert_rh_y_up_to_z_up(vectors: np.array) -> np.array:
    """ Convert right handed y up system to right handed z up.

    This is done by rotating around the x axis, or equivalent to:
        float y = position.Y;
        position.Y = -position.Z;
        position.Z = y;

    Args:
        vectors --  data in rh y up system (nframes, 3)

    """
    vectors = vectors[:, [0, 2, 1]]
    vectors[:, 1] *= -1
    return vectors

def convert_rh_y_down_to_z_up(vectors: np.array) -> np.array:
    """ Convert right handed y down system to right handed z up.

    This is done by rotating around the x axis, or equivalent to:
        float z = position.Z;
        position.Z = -position.Y;
        position.Y = z;

    Args:
        vectors --  data in rh -y up system (nframes, 3)

    """
    vectors = vectors[:, [0, 2, 1]]
    vectors[:, 2] *= -1
    return vectors


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

    @classmethod
    def feature_fields(cls) -> List[str]:
        """ Labels for the feature columns """
        return ["body_qw", "body_qx", "body_qy", "body_qz", "body_tx",
                "body_ty", "body_tz", "head_qw", "head_qx", "head_qy",
                "head_qz", "head_tx", "head_ty", "head_tz", "speaking"]

    def extract(self, body: Dict, speech: Dict) -> np.ndarray:
        """ Compute feature array for a subject """
        # Assert that all frames are valid
        assert body["bValid"]
        # body pose shape (nframes, 7), consisting of (quaternion, location)
        body_normal = convert_rh_y_down_to_z_up(body["bodyNormal"].transpose())
        body_quat = self.compute_orientation(body_normal)
        neck_loc = body["joints19"][0:3, :].transpose() # (nframes, 3)
        neck_loc = convert_rh_y_down_to_z_up(neck_loc) # (nframes, 3)
        # head pose shape (nframes, 7)
        head_normal = convert_rh_y_down_to_z_up(body["faceNormal"].transpose())
        head_quat = self.compute_orientation(head_normal)
        nose_loc = body["joints19"][3:6, :].transpose() # (nframes, 3)
        nose_loc = convert_rh_y_down_to_z_up(nose_loc)
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

    def compute_orientation(self, normal: np.ndarray) -> np.ndarray:
        """ Compute the unit quaternion orientation from the normal direction.

        The quaternion in the first frame is constrained to the positive real
        hemisphere. Additionally, the shortest path rotation is ensured between
        each frame and the next one.

        Args:
            normal  --  The forward direction, shape (nframes, 3)

        """
        # Assume people start facing -Y with Z up. Mainly so that to_track_quat
        # works as expected, and a side-effect bonus for help with viz
        # This is crucial. Make sure data created in OpenGL (RH Y Up), is
        # converted to Blender's coordinate system (RH Z Up)
        quats = [Vector(v).to_track_quat("-Y", "Z") for v in normal]
        # Constraint the first quaternion to the positive real hemisphere
        # for consistency
        if quats[0].w < 0:
            quats[0].negate()
        # Ensure the shortest path from each frame to the next
        for i in range(len(quats)-1):
            if quats[i].dot(quats[i+1]) < 0:
                quats[i+1].negate()
        return np.array(quats)


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

    # Add columns for frame, group_id, group_size, and person_id
    frames = np.arange(body["startFrame"], body["startFrame"] + seq_len)
    frames = np.repeat(frames, gsize)
    gids = [group_id] * seq_len * gsize
    gsizes = [gsize] * seq_len * gsize
    person_ids = [sub["humanId"] for sub in body["subjects"]]
    pids = np.tile(person_ids, seq_len)

    # Construct the group dataframe
    info_df = pd.DataFrame(
        {
            "frame": frames,
            "group_id": gids,
            "group_size": gsizes,
            "person_id": pids
        }
    )
    data_df = pd.DataFrame(interleaved.astype(np.float32),
                           columns=extractor.feature_fields())

    return pd.concat([info_df, data_df], axis=1)
