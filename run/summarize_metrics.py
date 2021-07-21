#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: summarize_metrics.py
# Created Date: Tuesday, January 26th 2021, 1:50:43 pm
# Author: Chirag Raman
#
# Copyright (c) 2021 Chirag Raman
###

import argparse
from pathlib import Path

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from lightning.callbacks import MetricsComputer


def main() -> None:
    """ Entry point for the program """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--metrics_path", type=str,
                        help="path to the metrics dataframe")
    parser.add_argument("--nposes", type=int, default=2,
                        help="number of poses in the data")
    args = parser.parse_args()

    # Load test metrics and log summary
    metrics = pd.read_hdf(args.metrics_path)
    out_dir = Path(args.metrics_path).parents[0]

    # Log the dataframe description of the summary metrics to csv
    summary_metrics = metrics[MetricsComputer.summary_columns(args.nposes)]
    summary_metrics = summary_metrics.apply(pd.to_numeric)
    stats = summary_metrics.describe()
    stats[~stats.index.isin(["25%", "50%", "75%"])].to_csv(
        out_dir/"summary_metrics.csv", index_label="statistic"
    )

if __name__ == "__main__":
    main()
