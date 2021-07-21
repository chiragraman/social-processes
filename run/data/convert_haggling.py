#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: convert_haggling.py
# Created Date: Thursday, January 1st 1970, 1:00:00 am
# Author: Chirag Raman
#
# Copyright (c) 2020 Chirag Raman
###


import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from constants import paths
from data.preprocessing import convert_panoptic_group
from data.preprocessing import PanopticBasicFeatures


# Test group names from https://github.com/CMU-Perceptual-Computing-Lab/ssp
TEST_SESSIONS = [
    "170221_haggling_b1", "170221_haggling_b2", "170221_haggling_b3",
    "170228_haggling_b1", "170228_haggling_b2", "170228_haggling_b3"
]

# Whitelisted groups from https://github.com/CMU-Perceptual-Computing-Lab/
# socialSignalPred/blob/7b209e41d0efac4b9f990ffed04653eb25c97cf1/
# motionsynth_data/data/processed/export_panopticDB_faceSpeech_group.py#L246
WHITELISTED_GROUPS = {
    "170221_haggling_b1_group0", "170221_haggling_b1_group2",
    "170221_haggling_b1_group3", "170221_haggling_b1_group4",
    "170221_haggling_b2_group1", "170221_haggling_b2_group2",
    "170221_haggling_b2_group4", "170221_haggling_b2_group5",
    "170221_haggling_b3_group0", "170221_haggling_b3_group1",
    "170221_haggling_b3_group2", "170228_haggling_b1_group0",
    "170228_haggling_b1_group1", "170228_haggling_b1_group2",
    "170228_haggling_b1_group3", "170228_haggling_b1_group6",
    "170228_haggling_b1_group7", "170228_haggling_b1_group8",
    "170228_haggling_b1_group9", "170221_haggling_m1_group0",
    "170221_haggling_m1_group2", "170221_haggling_m1_group3",
    "170221_haggling_m1_group4", "170221_haggling_m1_group5",
    "170221_haggling_m2_group2", "170221_haggling_m2_group3",
    "170221_haggling_m2_group5", "170221_haggling_m3_group0",
    "170221_haggling_m3_group1", "170221_haggling_m3_group2",
    "170224_haggling_a1_group0", "170224_haggling_a1_group1",
    "170224_haggling_a1_group3", "170224_haggling_a1_group4",
    "170224_haggling_a1_group5", "170224_haggling_a1_group6",
    "170224_haggling_a2_group0", "170224_haggling_a2_group1",
    "170224_haggling_a2_group2", "170224_haggling_a2_group6",
    "170224_haggling_a3_group0", "170224_haggling_b1_group0",
    "170224_haggling_b1_group4", "170224_haggling_b1_group5",
    "170224_haggling_b1_group6", "170224_haggling_b2_group0",
    "170224_haggling_b2_group1", "170224_haggling_b2_group4",
    "170224_haggling_b2_group5", "170224_haggling_b2_group7",
    "170224_haggling_b3_group0", "170224_haggling_b3_group2",
    "170228_haggling_a1_group0", "170228_haggling_a1_group1",
    "170228_haggling_a1_group4", "170228_haggling_a1_group6",
    "170228_haggling_a2_group0", "170228_haggling_a2_group1",
    "170228_haggling_a2_group2", "170228_haggling_a2_group4",
    "170228_haggling_a2_group5", "170228_haggling_a2_group6",
    "170228_haggling_a2_group7", "170228_haggling_a3_group1",
    "170228_haggling_a3_group2", "170228_haggling_a3_group3",
    "170228_haggling_b2_group0", "170228_haggling_b2_group1",
    "170228_haggling_b2_group4", "170228_haggling_b2_group5",
    "170228_haggling_b2_group8", "170228_haggling_b3_group0",
    "170228_haggling_b3_group1", "170228_haggling_b3_group2",
    "170228_haggling_b3_group3", "170404_haggling_a1_group2",
    "170404_haggling_a2_group1", "170404_haggling_a2_group2",
    "170404_haggling_a2_group3", "170404_haggling_a3_group0",
    "170404_haggling_a3_group1", "170404_haggling_b1_group3",
    "170404_haggling_b1_group6", "170404_haggling_b1_group7",
    "170404_haggling_b2_group1", "170404_haggling_b2_group4",
    "170404_haggling_b2_group6", "170404_haggling_b3_group1",
    "170404_haggling_b3_group2", "170407_haggling_a1_group1",
    "170407_haggling_a1_group3", "170407_haggling_a1_group5",
    "170407_haggling_a2_group3", "170407_haggling_a2_group5",
    "170407_haggling_b1_group0", "170407_haggling_b1_group1",
    "170407_haggling_b1_group2", "170407_haggling_b1_group3",
    "170407_haggling_b1_group4", "170407_haggling_b1_group6",
    "170407_haggling_b1_group7", "170407_haggling_b2_group0",
    "170407_haggling_b2_group1", "170407_haggling_b2_group2",
    "170407_haggling_b2_group4", "170407_haggling_b2_group5",
    "170407_haggling_b2_group6"
}

def load_group(path: Path):
    """ Load the group pkl file specified at the path """
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


def main():
    """ Convert each group in the panoptic haggling dataset to pd.DataFrame """
    parser = argparse.ArgumentParser()
    parser.add_argument("--haggling_dir", type=str,
                        help="root input directory for the haggling dataset")
    parser.add_argument("--outfile", type=str,
                        help="hdf5 filename to store processed DataFrames")
    args = parser.parse_args()

    # Create the output dir
    inroot = Path(args.haggling_dir)
    outdir = inroot / "processed"
    outdir.mkdir(parents=True, exist_ok=True)

    # Get names of groups in the input body and speech dir and assert that
    # they match
    body_dir = inroot / paths.PANOPTIC_BODY_DIR
    speech_dir = inroot / paths.PANOPTIC_SPEECH_DIR
    groups = [g.name for g in body_dir.glob("*.pkl")]
    speech_groups = [g.name for g in speech_dir.glob("*.pkl")]
    assert(groups == speech_groups), (
        "Body and Speech group names must match"
    )

    # Convert the pkl data in the body and speech directories to a unified
    # DataFrame and write to the outdir
    extractor = PanopticBasicFeatures()
    dfs = []
    for g in groups:
        print("Processing: ", g)
        body_pkl = load_group(body_dir/g)
        speech_pkl = load_group(speech_dir/g)
        group_df = convert_panoptic_group(
            Path(g).stem, [body_pkl, speech_pkl], extractor
        )
        dfs.append(group_df)

    # Concatenate all groups into a master dataframe
    master_df = pd.concat(dfs, ignore_index=True)

    # Write the master dataframe to the outfile
    master_df.to_hdf(outdir/args.outfile, key="haggling")
    print("Memory Usage: {} MB ".format(
         master_df.memory_usage(deep=True).sum()/1000000))
    print(master_df.head())

    # Write the group names to "train.csv" and "test.csv"
    train_groups = [Path(g).stem for g in groups if not any(s in g for s in TEST_SESSIONS)]
    train_groups = list(set(train_groups).intersection(WHITELISTED_GROUPS))
    test_groups = [Path(g).stem for g in groups if Path(g).stem not in train_groups]
    test_groups = list(set(test_groups).intersection(WHITELISTED_GROUPS))
    with open(outdir/"train.txt", "w") as f:
        print(*train_groups, sep="\n", file=f)
    with open(outdir/"test.txt", "w") as f:
        print(*test_groups, sep="\n", file=f)

    # Generate 5 folds for train/val splits
    kf = KFold(n_splits=5)
    train_groups = np.array(train_groups)
    for i, (train_idx, val_idx) in enumerate(kf.split(train_groups)):
        # Write the keys to a file for each fold
        with open(outdir/"fold{}.pkl".format(i), "wb") as f:
            pickle.dump(
                {"train": train_groups[train_idx], "val":train_groups[val_idx]},
                f
            )

    # Write the summary of the train groups for standardizing test set
    train_df = master_df.loc[master_df.group_id.isin(train_groups)]
    train_df.describe().to_hdf(outdir/"train_description.h5", key="haggling_train")


if __name__ == "__main__":
    main()