#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: toy_sine.py
# Created Date: Monday, November 11th 2019, 5:46:17 pm
# Author: Chirag Raman
#
# Copyright (c) 2019 Chirag Raman
###


import numpy as np


def sin_and_stop_for_phase(seq_len: int, phase: float) -> np.ndarray:
    """ Same as `sin_and_stop`, except accepts phase

    Args:
        seq_len : The number of time steps in the generated signal

    Returns the (seq_len, 2, 1) dim np.ndarray

    """
    sin = np.sin(np.linspace(0 * np.pi + phase, 3 * np.pi + phase, seq_len))
    sin = sin[:, np.newaxis]
    timesteps_to_fix = (seq_len // 4) + 1
    sin_stopped = sin.copy()
    sin_stopped[-timesteps_to_fix:] = sin_stopped[-timesteps_to_fix]
    return np.stack((sin, sin_stopped), axis=1)