#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import matplotlib.pyplot as plt
import datetime, time
import tensorflow as tf

def initialize(cfg, state):

    # Initialize the time with starting time
    state.t = tf.Variable(float(cfg.processes.dummy_time.start))

    state.itsave = -1

    state.dt = tf.Variable(float(cfg.processes.dummy_time.step_max))

    state.dt_target = tf.Variable(float(cfg.processes.dummy_time.step_max))

    state.time_save = np.ndarray.tolist(
        np.arange(cfg.processes.dummy_time.start, cfg.processes.dummy_time.end, cfg.processes.dummy_time.save)
    ) + [cfg.processes.dummy_time.end]

    state.time_save = tf.constant(state.time_save, dtype="float32")

def update(cfg, state):
    if hasattr(state, "logger"):
        state.logger.info(
            "Update DT at time : " + str(state.t.numpy())
        )


    state.dt = state.dt_target

    # modify dt such that times of requested savings are reached exactly
    if state.time_save[state.itsave + 1] <= state.t + state.dt:
        state.dt = state.time_save[state.itsave + 1] - state.t
        state.saveresult = True
        state.itsave += 1
    else:
        state.saveresult = False 
        # make a tensor out of state.dt
        state.dt = state.t/state.t

    # the first loop is not advancing
    if state.it >= 0:
        state.t.assign(state.t + state.dt)

    state.continue_run = (state.t < cfg.processes.dummy_time.end)

def finalize(cfg, state):
    pass
