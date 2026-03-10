#!/usr/bin/env python3

# Copyright (C) 2023 Andreas Henz (andreas.henz@geo.uzh.ch)
# Published under the GNU GPL (Version 3), check at the LICENSE file

# This script only works with write ncdf, not with write tif module
# October 2023
# updated 27. June 2024

import time
import tensorflow as tf

# from igm.modules.utils import *


def initialize(cfg, state):
    state.tcomp_write_icemargin = []
    
    state.tlast_write_icemargin = tf.Variable(cfg.processes.heatmaps.time_start_icemargin)
    
    # instead of initializing with zeros, we initialize with a number of years between 0 and 1 zero
    # at the highest peaks and 1 in the valleys based on state.topg
    # normalize topg to 0-1
    topg_norm = (state.topg - tf.reduce_min(state.topg)) / (tf.reduce_max(state.topg) - tf.reduce_min(state.topg))
    # invert topg_norm
    # topg_norm = 1 - topg_norm
    
 
    # Initialize grid with zeros
    state.icemargin = tf.Variable(tf.zeros_like(state.topg))
    
    # Initialize grid with deglaciation age (to make gradients work, start with topg_norm)
    state.lasticecover = tf.Variable(topg_norm)  # Fill with starting time + topg_norm for gradients
    # Example of using NaN in TensorFlow:
    # tf.constant(float('nan')) or tf.constant(np.nan)
    # tf.experimental.numpy.nan is also available
    # For a tensor filled with NaNs:
    # tf.fill(state.topg.shape, tf.constant(float('nan')))
     
    # ice free cumulative age (to make gradients work, start with topg_norm)
    state.icefree_cummulative = tf.Variable(1-topg_norm)
    
    # Add saving of heatmaps to ncdf ex file:
    cfg.outputs.write_ncdf.vars_to_save.append("icemargin")
    cfg.outputs.write_ncdf.vars_to_save.append("lasticecover")
    cfg.outputs.write_ncdf.vars_to_save.append("icefree_cummulative")

    state.var_info_ncdf_ex["icemargin"] = ["Ice margin age", "years"]
    state.var_info_ncdf_ex["lasticecover"] = ["Ice cover age", "years"]
    state.var_info_ncdf_ex["icefree_cummulative"] = ["Ice free cumulative age", "years"]

def update(cfg, state):
    # update smb each X years
    if (state.t - state.tlast_write_icemargin) >= cfg.processes.heatmaps.update_freq:
        
        dt = state.t - state.tlast_write_icemargin # dt is not perfectly 1.0, so we need to use the real dt
        
        # for time profiling
        state.tcomp_write_icemargin.append(time.time())

        # for ice margin, errase previous ice margins that are in the accumulation area
        condition = tf.logical_and(state.thk > cfg.processes.heatmaps.min_thickness, state.smb > 0.2)
        indices = tf.where(condition)
        updates = tf.zeros(tf.shape(indices)[0], dtype=tf.float32)
        state.icemargin.assign(tf.tensor_scatter_nd_update(state.icemargin, indices, updates))
        
        # add dt if thickness is in certain range
        condition = tf.logical_and(state.thk > cfg.processes.heatmaps.min_thickness, state.thk < cfg.processes.heatmaps.max_thickness)
        # condition 2 is below where the smb is negative
        condition = tf.logical_and(condition, state.smb < 0.2)
        indices = tf.where(condition)
        updates = tf.ones(tf.shape(indices)[0], dtype=tf.float32) * dt
        state.icemargin.assign(tf.tensor_scatter_nd_add(state.icemargin, indices, updates))

        # =================================
        # redefine last ice cover/last ice free age
        # where thickness > 0 (ice covered) but fresh, so last ice cover -0, set to zero
        indices = tf.where(tf.logical_and(state.thk > 0, state.lasticecover <= 0))
        updates = tf.zeros(tf.shape(indices)[0], dtype=tf.float32)
        state.lasticecover.assign(tf.tensor_scatter_nd_update(state.lasticecover, indices, updates))

        # where thickness <= 0 (ice free) but last ice cover < 0, set to 0
        indices = tf.where(tf.logical_and(state.thk <= 0, state.lasticecover > 0))
        updates = tf.zeros(tf.shape(indices)[0], dtype=tf.float32)
        state.lasticecover.assign(tf.tensor_scatter_nd_update(state.lasticecover, indices, updates))

        # where thickness > 0 (ice covered) and last ice cover >= 0, add the time step to last ice cover
        indices = tf.where(tf.logical_and(state.thk > 0, state.lasticecover >= 0))
        updates = tf.ones(tf.shape(indices)[0], dtype=tf.float32) * dt
        state.lasticecover.assign(tf.tensor_scatter_nd_add(state.lasticecover, indices, updates))

        # where thickness <= 0 (ice free) and last ice cover <= 0, add number of years ice free (minus dt)
        indices = tf.where(tf.logical_and(state.thk <= 0, state.lasticecover <= 0))
        updates = -tf.ones(tf.shape(indices)[0], dtype=tf.float32) * dt
        state.lasticecover.assign(tf.tensor_scatter_nd_add(state.lasticecover, indices, updates))

        
        # ice free cumulative age
        # where ice free and icefree_cummulative is >= 0, add 1
        indices = tf.where(state.thk <= 0)
        updates = tf.ones(tf.shape(indices)[0], dtype=tf.float32) * dt
        state.icefree_cummulative.assign(tf.tensor_scatter_nd_add(state.icefree_cummulative, indices, updates))

        # # where ice free and icefree_cummulative is < 0, set to 0
        # indices = tf.where(tf.logical_and(state.thk <= 0, state.icefree_cummulative < 0))
        # updates = tf.zeros(tf.shape(indices)[0], dtype=tf.float32)
        # state.icefree_cummulative.assign(tf.tensor_scatter_nd_update(state.icefree_cummulative, indices, updates))
        # # where ice cover and icefree_cummulative is < 0, minus 1
        # indices = tf.where(tf.logical_and(state.thk > 0, state.icefree_cummulative <= 0))
        # updates = -tf.ones(tf.shape(indices)[0], dtype=tf.float32) * dt
        # state.icefree_cummulative.assign(tf.tensor_scatter_nd_add(state.icefree_cummulative, indices, updates))
        # # where ice cover and icefree_cummulative is >= 0, set to 0
        # indices = tf.where(tf.logical_and(state.thk > 0, state.icefree_cummulative > 0))
        # updates = tf.zeros(tf.shape(indices)[0], dtype=tf.float32)
        # state.icefree_cummulative.assign(tf.tensor_scatter_nd_update(state.icefree_cummulative, indices, updates))
        
        state.tlast_write_icemargin.assign(state.t)
        
        # for time profiling
        state.tcomp_write_icemargin[-1] -= time.time()
        state.tcomp_write_icemargin[-1] *= -1


def finalize(cfg, state):
       
    pass
