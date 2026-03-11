#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os
from netCDF4 import Dataset
import tensorflow as tf


def initialize(cfg, state):

    state.var_info_ncdf_ts = {}
    state.var_info_ncdf_ts["vol"] = ["Ice volume", "km^3"]
    state.var_info_ncdf_ts["area"] = ["Glaciated area", "km^2"]
    state.var_info_ncdf_ts["gi"] = ["glacial index", "unitless"]
    state.var_info_ncdf_ts["ela"] = ["median ELA of the region", "m a.s.l."]
    state.var_info_ncdf_ts["tas"] = ["Mean surface temperature at 1000 m elevation", "°C"]
    state.var_info_ncdf_ts["tss"] = ["Mean summer surface temperature at 1000 m elevation", "°C"]
    state.var_info_ncdf_ts["pr"] = ["Mean precipitation", "mm/yr"]

    print("Use of my custom write_ts process")
    print(state.var_info_ncdf_ts)


def run(cfg, state):
    if state.saveresult:
        vol = np.sum(state.thk) * (state.dx**2) / 10**9
        area = np.sum(state.thk > 1) * (state.dx**2) / 10**6
        if hasattr(state, "glacial_index"):
            gi = state.glacial_index
        else:
            gi = tf.Variable(np.nan, dtype=tf.float32)
        if hasattr(state, "precipitation"):
            pr = tf.reduce_mean(state.precipitation) * cfg.processes.smb_enhanced_accpdd.precip_offset
        else:
            pr = tf.constant(np.nan, dtype=tf.float32)
        if hasattr(state, "air_temp"):
            # mean lapse rate
            LR = tf.reduce_mean(state.LR, axis=(1,2))
            # adjust air temp to 1000 m elevation
            air_temp_1000m = tf.reduce_mean(state.air_temp, axis=(1,2)) + LR * (1000.0 - tf.reduce_mean(state.tempsurfref_snap)) 

            tas = tf.reduce_mean(air_temp_1000m)
            tss = tf.reduce_mean(air_temp_1000m[5:8])
        else:
            tas = tf.constant(np.nan, dtype=tf.float32)
            tss = tf.constant(np.nan, dtype=tf.float32)

        # produce the ela output (also without direct ela varaible, but from smb)
        if hasattr(state, "ela"):
            ela = np.percentile(state.ela, 50.0)
        else:
            # take smb to calculate ela
            smb = state.smb
            # Find indices where the absolute value of smb is less than 0.1
            smb_around_0 = np.abs(smb) < 0.1

            # Filter state.topg using the boolean mask
            filtered_topg = state.topg[smb_around_0]

        # Check if the filtered array is empty
        if len(filtered_topg) == 0 or np.isnan(filtered_topg).all():
            ela = np.nan
            print("No elements in smb are within 0.1 of zero.")
        else:
            # Compute the median of the filtered values
            ela = np.percentile(filtered_topg, 50.0)  # 50th percentile is the median
            
        # if more than 2/3 are in the accumulation are, that is problematic
        if np.sum(smb > 0) > 2/3 * smb.shape[0]*smb.shape[1]:
            print("Warning: More than 2/3 of the smb values are positive.")

        # make ela tensorflow variable
        ela = tf.Variable(ela, dtype=tf.float32)

        if not hasattr(state, "already_called_update_write_ts"):
            state.already_called_update_write_ts = True

            if hasattr(state, "logger"):
                state.logger.info("Initialize NCDF ts output Files")

            nc = Dataset( cfg.outputs.write_ts_henz.output_file,"w", format="NETCDF4" )

            nc.createDimension("time", None)
            E = nc.createVariable("time", np.dtype("float32").char, ("time",))
            E.units = "yr"
            E.long_name = "time"
            E.axis = "T"
            E[0] = state.t.numpy()

            for var in ["vol", "area", "gi", "ela", "tas", "tss", "pr"]:
                # print(var, state.var_info_ncdf_ts[var])
                E = nc.createVariable(var, np.dtype("float32").char, ("time"))
                E[0] = vars()[var].numpy()
                E.long_name = state.var_info_ncdf_ts[var][0]
                E.units = state.var_info_ncdf_ts[var][1]
            nc.close()

        else:
            if hasattr(state, "logger"):
                state.logger.info(
                    "Write NCDF ts file at time : " + str(state.t.numpy())
                )

            nc = Dataset( cfg.outputs.write_ts_henz.output_file, "a", format="NETCDF4" )
            d = nc.variables["time"][:].shape[0]

            nc.variables["time"][d] = state.t.numpy()
            for var in ["vol", "area", "gi", "ela", "tas", "tss", "pr"]:
                nc.variables[var][d] = vars()[var].numpy()
            nc.close()

