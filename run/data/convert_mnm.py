#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: convert_mnm.py
# Created Date: Wednesday, October 20th 2021, 12:21:27 pm
# Author: Chirag Raman
#
# Copyright (c) 2021 Chirag Raman
###


import argparse
import pickle
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from constants import paths
from data.preprocessing import (
    convert_mnm_group, FeatureExtractorInterface, MNMBasicFeatures
)


# Constants
GROUPS_DIR = "additional/FF_extended"
POSE_DIR = "additional/pose"
VIDEO_FPS = 20
RES_Y = 540
MIN_GROUP_DURATION = 20
FRAME_COL = "frame"
FRAME_START_COL = "frame_start"
FRAME_END_COL = "frame_end"
LEAVE_COL = "leaving"
ACTION_COL = "action"
SPEAKING_COL = "speaking"
OCCLUDED_COL = "occluded"
DAY_COL = "day"
CAM_COL = "cameraNum"
FRAMEID_COL = "frameId"
PERSON_COL = "person_id"
GROUP_COL = "group_id"
SIZE_COL = "group_size"
PARTICIPANTS_COLS = [DAY_COL, PERSON_COL]
N_ACTIONS = 9
ACTION_COLS = ["walking", "stepping", "drinking", SPEAKING_COL, "hand_gesture",
               "head_gesture", "laugh", "hair_touching", OCCLUDED_COL]

# Book-keeping
NFULL = []
NFULL_CHUNKS = 0
NPARTIAL = []
NPARTIAL_CHUNKS = 0

# Splits
TRAIN_DAYS = [2, 3]
TEST_DAYS = [1]


def load_groups(root_dir: Path) -> pd.DataFrame:
    """ Load the F-formation annotations into a dataframe

    Args:
        root_dir    : The directory containing the day-wise csv's

    Returns:
        The dataframe containing annotations from all days. The index of the
        dataframe is an added column "day" with values <1/2/3>. Annotations
        for a certain day (eg. Day 1) can then be accessed with `df.loc[<day>,:]`.
        Expects the day wise annotation file stem to end with the day number.

    """
    day_annotations = []
    for day_file in root_dir.glob("*.csv"):
        day_anno = pd.read_csv(day_file)
        day_anno[DAY_COL] = int(day_file.stem[-1])
        day_annotations.append(day_anno)
    groups = pd.concat(day_annotations)
    # Add columns for leaving anotations, frame start & end, and group id
    groups[LEAVE_COL] = groups.apply(
        lambda row: int(row["binary_label"].lower() == "leave"),
        axis=1
    )
    groups[FRAME_START_COL] = groups.apply(
        lambda row: row["time_start_secs"] * VIDEO_FPS,
        axis=1
    )
    groups[FRAME_END_COL] = groups.apply(
        lambda row: row["time_end_secs"] * VIDEO_FPS,
        axis=1
    )
    groups[GROUP_COL] = (groups[DAY_COL].astype(str) + "_" + groups["subjects"]).astype("category").cat.codes
    groups[SIZE_COL] = groups["subjects"].str.split().apply(len)
    return groups


def load_participant_ids(path: Path) -> pd.DataFrame:
    """ Return the dataframe for the participants' global ids """
    df = pd.read_csv(path, names=PARTICIPANTS_COLS)
    return df


def load_actions(path: Path, lost_path: Path, p_ids: pd.DataFrame):
    """ Load the manual social action annotations and add hierarchical indices

    Args:
        path        -- The path to the annotations csv
        lost_path   -- The path to the lost csv denoting participants
                       outside the camera FOV
        p_ids       -- The path to the participants id

    Returns:
        The annotations data frame with a hierarchical column index
        ["day", "participant", "action"] where "action" corresponds to the
        nine social action labels. Example access usage -
        actions.loc[:, pd.IndexSlice[day, [p_ids], [<action names>]]]
    """
    # Read all required csvs
    actions = pd.read_csv(path, header=None)
    lost = pd.read_csv(lost_path, header=None)

    # Construct and set index
    days = np.repeat(p_ids[DAY_COL].values, N_ACTIONS)
    ps = np.repeat(p_ids[PERSON_COL].values, N_ACTIONS)
    acs = np.tile(ACTION_COLS, len(p_ids))
    index = pd.MultiIndex.from_arrays(
        [days, ps, acs],
        names=(DAY_COL, PERSON_COL, ACTION_COL)
    )
    actions.columns = index

    # Set entries of lost participants to nan
    repeated_lost = pd.DataFrame(
        np.repeat(lost.values, N_ACTIONS, axis=1)
    )
    mask = (repeated_lost[repeated_lost.columns] != 1)
    mask.columns = actions.columns
    actions = actions.where(mask, other=np.nan)

    # Extract speaking and occluded columns
    actions = actions.loc[:, pd.IndexSlice[:, :, [SPEAKING_COL, OCCLUDED_COL]]]

    return actions


def process_group(
        group_row: Iterable[Tuple[Any, ...]], p_ids: pd.DataFrame,
        pose_root: Path, extractor: FeatureExtractorInterface
    ) -> Optional[pd.DataFrame]:
    """ Compute the processed features for the conversation group """
    group_df = None
    day = group_row.day
    f_start = group_row.frame_start
    f_end = group_row.frame_end
    participants = [int(p) for p in group_row.subjects.strip().split(" ")]

    if len(participants) > 1: # Skip singleton groups
        group_data = dict() # Map participant to features
        for p_id in participants:
            # Load pose features
            file_num = (p_ids.loc[(p_ids[DAY_COL] == day) & (p_ids[PERSON_COL] == p_id)].index[0]
                        - p_ids[p_ids[DAY_COL] == day].index[0] + 1)
            features_fname = pose_root / f"day{day}" / f"person{file_num}.csv"
            pose_features = pd.read_csv(features_fname)
            # Transform so that clockwise rotation is positive
            # and y is positive up in image plane
            pose_features[["hOrient_deg", "bOrient_deg"]] *= -1
            pose_features[["sh1Y", "headY", "sh2Y", "gazeY"]] = (
                RES_Y -  pose_features[["sh1Y", "headY", "sh2Y", "gazeY"]]
            )
            # Get the slice for the features
            pose_slice = pose_features.loc[pose_features.frameId.between(f_start, f_end)]
            # Store features for person id using "<day>_<p_id>" as key
            group_data[f"{day}_{p_id}"] = pose_slice

        # Combine features into single dataframe and drop rows that have nan
        keys, values = zip(*group_data.items())
        group_raw = pd.concat(values, axis=1, keys=keys)
        group_raw = group_raw.dropna()

        # Get processed features expected by SocialDataset
        if len(group_raw) != 0:
            # Check if visible in single cam
            cam_values = group_raw.loc[:, pd.IndexSlice[:, CAM_COL]].values
            ncams = len(np.unique(cam_values))
            if ncams == 1:
                # Pass raw features into group converter to obtain transformed
                # final features for every chunk of contiguous frames
                p_id0 = keys[0]
                predicate = (group_raw[(p_id0, FRAMEID_COL)] - group_raw[(p_id0, FRAMEID_COL)].shift() != VIDEO_FPS)
                group_chunks = []
                for _, g_raw in group_raw.groupby(predicate.cumsum()):
                    if len(g_raw) >= MIN_GROUP_DURATION:
                        chunk_df = convert_mnm_group(
                            f"{day}_{group_row.group_id}", group_row.group_size,
                            g_raw, extractor
                        )
                        group_chunks.append(chunk_df)
                if len(group_chunks) > 0:
                    group_df = pd.concat(group_chunks)
                    global NFULL, NFULL_CHUNKS
                    NFULL.append(f"{day}_{group_row.group_id}")
                    NFULL_CHUNKS += len(group_chunks)

            # Check if shorter slices are visible in a single cam
            if ncams != 1:
                # Count contiguous sequences where full group is in single cam
                frames_cams_slice = group_raw.loc[:, pd.IndexSlice[:, [FRAMEID_COL, CAM_COL]]]
                single_cams = frames_cams_slice[
                    frames_cams_slice.loc[:, pd.IndexSlice[:, CAM_COL]].nunique(axis=1) == 1
                ]
                p_id0 = keys[0]
                predicate = (
                    (single_cams[(p_id0, CAM_COL)].shift() != single_cams[(p_id0, CAM_COL)])
                    | (single_cams[(p_id0, FRAMEID_COL)] - single_cams[(p_id0, FRAMEID_COL)].shift() != VIDEO_FPS)
                )

                # Pass contiguous chunk raw features into group converter to
                # obtain transformed final features
                group_chunks = []
                for _, g_raw in single_cams.groupby(predicate.cumsum()):
                    if len(g_raw) >= MIN_GROUP_DURATION:
                        g_raw_features = group_raw.loc[g_raw.index, :]
                        chunk_df = convert_mnm_group(
                            f"{day}_{group_row.group_id}", group_row.group_size,
                            g_raw_features, extractor
                        )
                        group_chunks.append(chunk_df)
                if len(group_chunks) > 0:
                    group_df = pd.concat(group_chunks)
                    global NPARTIAL, NPARTIAL_CHUNKS
                    NPARTIAL.append(f"{day}_{group_row.group_id}")
                    NPARTIAL_CHUNKS += len(group_chunks)

    return group_df


def process_mnm(groups_df: pd.DataFrame, p_ids: pd.DataFrame,
                pose_root: Path) -> pd.DataFrame:
    """ Load the F-formation annotations and get features for each group """
    extractor = MNMBasicFeatures()
    groups = []
    for day in groups_df[DAY_COL].unique():
        for group_row in groups_df.loc[groups_df[DAY_COL] == day].itertuples():
            group_df = process_group(group_row, p_ids, pose_root, extractor)
            if group_df is not None:
                groups.append(group_df)
    main_df = pd.concat(groups)
    return main_df


def main():
    """ Convert each group in the panoptic haggling dataset to pd.DataFrame """
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str,
                        help="root annotations directory for the MatchNMingle dataset")
    parser.add_argument("--outfile", type=str,
                        help="hdf5 filename to store processed DataFrames")
    args = parser.parse_args()

    # Create the output dir
    inroot = Path(args.root_dir)
    outdir = inroot / "processed"
    outdir.mkdir(parents=True, exist_ok=True)

    # Process annotations
    groups_df = load_groups(inroot / GROUPS_DIR)
    groups_df.to_csv(outdir / "groups.csv", index=False)
    p_ids = load_participant_ids(inroot / paths.MNM_PARTICIPANTS)
    main_df = process_mnm(groups_df, p_ids, inroot / POSE_DIR)

    # Write the master dataframe to the outfile
    main_df.to_hdf(outdir/args.outfile, key="mingle")
    print("Memory Usage: {} MB ".format(
          main_df.memory_usage(deep=True).sum()/1000000))

    print(f"N groups: total unique = {len(set(NFULL) | set(NPARTIAL))} ; "
          f"visiblity (fully/partial) = {len(set(NFULL))}/{len(set(NPARTIAL))}; "
          f"N occurrences (full / partial) = {len(NFULL)}/{len(NPARTIAL)}; "
          f"N chunks (full / partial) = {NFULL_CHUNKS}/{NPARTIAL_CHUNKS}")
    print("Fully visible groups: \n", NFULL)
    print("\nPartially visible groups: \n", NPARTIAL)

    # Write the train and test group ids
    train_groups = main_df[main_df[GROUP_COL].str[0].astype(int).isin(TRAIN_DAYS)][GROUP_COL].unique()
    np.savetxt(outdir/"train.txt", train_groups, fmt="%s")
    test_groups = main_df[main_df[GROUP_COL].str[0].astype(int).isin(TEST_DAYS)][GROUP_COL].unique()
    np.savetxt(outdir/"test.txt", test_groups, fmt="%s")

    # Generate 5 folds for train/val splits
    kf = KFold(n_splits=5)
    for i, (train_idx, val_idx) in enumerate(kf.split(train_groups)):
        # Write the keys to a file for each fold
        with open(outdir/"fold{}.pkl".format(i), "wb") as f:
            pickle.dump(
                {"train": train_groups[train_idx], "val":train_groups[val_idx]},
                f
            )

    # Write the summary of the train groups for standardizing test set
    train_df = main_df.loc[main_df[GROUP_COL].isin(train_groups)]
    train_df.describe().to_hdf(outdir/"train_description.h5", key="mnm_train")


if __name__ == "__main__":
    main()