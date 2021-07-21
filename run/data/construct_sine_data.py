#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: gen_toy_sine.py
# Created Date: Monday, November 11th 2019, 5:56:13 pm
# Author: Chirag Raman
#
# Copyright (c) 2019 Chirag Raman
###


from pathlib import Path
import argparse
import numpy as np

from data.signal import sin_and_stop_for_phase


def generate_phased_sin_with_stops(seq_len: int) -> np.ndarray:
    """ Generates a 1d sinusoidal signal with and without stops. Refer to
    data.signal.sin_and_stop() for details

    Returns the (seq_len, 2*n_samples, 1) np.ndarray
    """
    phases = np.arange(0, 2*np.pi, 0.001)
    samples = [sin_and_stop_for_phase(seq_len, phase) for phase in phases]
    data = np.concatenate(samples, axis=1).astype(np.float32)
    return data


def main():
    """ Generate the toy datasets for sinusoidal data """
    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", type=str, default="waves.npy",
                        help="filename to save the generated waves")
    parser.add_argument("--outdir", type=str, default="",
                        help="directory to store artefacts in")
    parser.add_argument("-s", "--seq_len", type=int, default=20,
                        help="number of time steps in the sequence")
    parser.add_argument("-n", "--n_samples", type=int, default=1000,
                        help="number of samples to generate")
    args = parser.parse_args()

    # Generate the head glance for all phases between 0 and 2pi
    data = generate_phased_sin_with_stops(args.seq_len)

    # Write to file
    outpath = Path(args.outdir) / args.outfile
    outpath.parent.mkdir(parents=True, exist_ok=True)
    np.save(outpath, data)


if __name__ == "__main__":
    main()
