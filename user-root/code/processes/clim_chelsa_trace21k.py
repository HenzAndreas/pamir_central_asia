#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

# This file is in the beginnning taken from the paleo alps example in IGM (Guillaume Jouvet)
# and adapted to the non-glacial index but transient climate output by CHELSA-Trace21k paleo climate dataset.
# 2025, Andreas Henz

import numpy as np
import os
import tensorflow as tf
import xarray as xr
from scipy.interpolate import RectBivariateSpline, interp1d
from igm.utils.math.interp1d_tf import interp1d_tf
 
def initialize(cfg, state):
    print("Use of TRace21k CHELSA climate data")

    check_all_folders_exist(cfg, state)

    # time frames
    if hasattr(cfg.processes, "time"):
        t_start = cfg.processes.time.start
    else:
        t_start = cfg.processes.dummy_time.start
    t_start = int((t_start // 100 - 1) * 100) # lower 100
    state.time_frames = tf.Variable([int(t_start), int(t_start + 100)], dtype=tf.float32)


    load_climate_data_chelsa_first_time(cfg, state)

    state.air_temp = tf.Variable(state.air_temp_snap[:, :, :, 0], dtype="float32")
    state.air_temp_sd = tf.Variable(state.air_temp_sd_snap[:, :, :, 0], dtype="float32")
    state.precipitation = tf.Variable(
        state.precipitation_snap[:, :, :, 0], dtype="float32"
    )
    state.tempsurfref = tf.Variable(state.tempsurfref_snap[:, :, 0], dtype="float32")
    state.LR = tf.Variable(state.LR_snap[: ,: ,: ,0], dtype="float32")

    # state.meanprec = tf.math.reduce_mean(state.precipitation, axis=0)
    # state.meantemp = tf.math.reduce_mean(state.air_temp, axis=0)

    state.tlast_clim = tf.Variable(-1.0e50, dtype="float32")

    plot_first_read_results(cfg, state)


def update(cfg, state):
    if (state.t - state.tlast_clim) >= cfg.processes.clim_chelsa_trace21k.update_freq:
        if hasattr(state, "logger"):
            state.logger.info("update climate at time : " + str(state.t.numpy()))

        # year rest to 100 years
        s = tf.abs(state.t % 100) / 100.0

        # update climate snapshots if state.t is in the next time frame
        if state.t >= state.time_frames[1]:
            state.time_frames.assign([state.time_frames[1], state.time_frames[1] + 100])

            # load new climate data
            load_climate_data_chelsa(cfg, state)

        state.LR.assign((1 - s) * state.LR_snap[:,:,:, 0] + s * state.LR_snap[:, :, :, 1])

        lapse_rate_cor = (
            state.tempsurfref_snap[:, :, 1] - state.tempsurfref_snap[:, :, 0]
            ) * (state.LR)
        air_temp_0_on_surf_1 = state.air_temp_snap[:, :, :, 0] + lapse_rate_cor

        state.air_temp.assign(
            (1 - s) * air_temp_0_on_surf_1 + s * state.air_temp_snap[:, :, :, 1]
        )
        state.precipitation.assign(
            (1 - s) * state.precipitation_snap[:, :, :, 0]
            + s * state.precipitation_snap[:, :, :, 1]
        )
        state.air_temp_sd.assign(
            (1 - s) * state.air_temp_sd_snap[:, :, :, 0]
            + s * state.air_temp_sd_snap[:, :, :, 1]
        )
        state.tempsurfref.assign(state.tempsurfref_snap[:, :, 1])

        lapse_rate_cor = (state.usurf - state.tempsurfref) * (state.LR)

        state.air_temp.assign(state.air_temp + lapse_rate_cor)

        # for debugging plot lapse rate correction
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # ax.imshow(np.mean(lapse_rate_cor, axis=0), cmap="coolwarm", origin='lower')
        # cbar = plt.colorbar(ax.images[0], ax=ax)
        # cbar.set_label("Lapse rate correction [°C]")
        # plt.title("Lapse rate correction")
        # plt.savefig(os.path.join(state.original_cwd, f"lapse_rate_correction_{int(state.t.numpy())}.png"))

        state.tlast_clim.assign(state.t)

        # plot_first_read_results(cfg, state)


def finalize(cfg, state):
    pass


############################################################################################################

def check_all_folders_exist(cfg, state):
    """
    check if all files exist
    """
    full_filenames = [
        os.path.join(state.original_cwd, cfg.processes.clim_chelsa_trace21k.dem_filename),
        os.path.join(state.original_cwd, cfg.processes.clim_chelsa_trace21k.vertical_lapse_rate_filename),
        os.path.join(state.original_cwd, cfg.processes.clim_chelsa_trace21k.precipitation_filename),
        os.path.join(state.original_cwd, cfg.processes.clim_chelsa_trace21k.temperature_filename)
    ]

    # check if all files exist
    for filename in full_filenames:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist.")

    # save only base names, without the time stamp
    for filename in full_filenames:
        # remove the last number after '_' in the filename
        if "_" in filename:
            temp_filename = "_".join(filename.rsplit("_", 1)[:-1])
        # save the base name without the time stamp for the relevant filename
        if filename.endswith(cfg.processes.clim_chelsa_trace21k.temperature_filename):
            cfg.processes.clim_chelsa_trace21k.temperature_filename = temp_filename
        elif filename.endswith(cfg.processes.clim_chelsa_trace21k.precipitation_filename):
            cfg.processes.clim_chelsa_trace21k.precipitation_filename = temp_filename
        elif filename.endswith(cfg.processes.clim_chelsa_trace21k.vertical_lapse_rate_filename):
            cfg.processes.clim_chelsa_trace21k.vertical_lapse_rate_filename = temp_filename
        elif filename.endswith(cfg.processes.clim_chelsa_trace21k.dem_filename):
            cfg.processes.clim_chelsa_trace21k.dem_filename = temp_filename

def load_climate_data_chelsa_first_time(cfg, state):
    """
    load the ncdf climate file containing precipitation, temperatures, lapse rate and surface elevation
    """

    # load the first snapshot of the climate data
    climate_snapshot = load_climate_data_one_snapshot(cfg, state, state.time_frames[0])

    # interpolate over a year
    climate_snapshot = interpolate_climate_over_a_year(
        [climate_snapshot], cfg.processes.clim_chelsa_trace21k.temporal_resampling
    )[0]

    # save the climate snapshot in the state
    state.air_temp_snap = np.expand_dims(climate_snapshot[0], axis=-1)
    state.air_temp_sd_snap = np.expand_dims(climate_snapshot[1], axis=-1)
    state.precipitation_snap = np.expand_dims(climate_snapshot[2], axis=-1)
    state.tempsurfref_snap = np.expand_dims(climate_snapshot[3], axis=-1)
    state.LR_snap = np.expand_dims(climate_snapshot[4], axis=-1)

def load_climate_data_chelsa(cfg, state):
    """
    load the ncdf climate file containing precipitation, temperatures, lapse rate and surface elevation
    """
    # overwrite the new snapshot with the old one from before and add the new one after
    climate_snapshot_0 = [state.air_temp_snap[..., -1], 
                          state.air_temp_sd_snap[..., -1], 
                          state.precipitation_snap[..., -1], 
                          state.tempsurfref_snap[..., -1], 
                          state.LR_snap[..., -1]]
    
    climate_snapshot_1 = load_climate_data_one_snapshot(
        cfg, state, state.time_frames[1])
    climate_snapshot = [climate_snapshot_0, climate_snapshot_1]

    ##############

    climate_snapshot = interpolate_climate_over_a_year(climate_snapshot,cfg.processes.clim_chelsa_trace21k.temporal_resampling)

    ##############

    state.air_temp_snap = tf.concat(
        [
            tf.expand_dims(tf.convert_to_tensor(climate_snapshot[0][0], dtype=tf.float32), axis=-1),
            tf.expand_dims(tf.convert_to_tensor(climate_snapshot[1][0], dtype=tf.float32), axis=-1),
        ],
        axis=-1,
    )
    state.air_temp_sd_snap = tf.concat(
        [
            tf.expand_dims(tf.convert_to_tensor(climate_snapshot[0][1], dtype=tf.float32), axis=-1),
            tf.expand_dims(tf.convert_to_tensor(climate_snapshot[1][1], dtype=tf.float32), axis=-1),
        ],
        axis=-1,
    )
    state.precipitation_snap = tf.concat(
        [
            tf.expand_dims(tf.convert_to_tensor(climate_snapshot[0][2], dtype=tf.float32), axis=-1),
            tf.expand_dims(tf.convert_to_tensor(climate_snapshot[1][2], dtype=tf.float32), axis=-1),
        ],
        axis=-1,
    )
    state.tempsurfref_snap = tf.concat(
        [
            tf.expand_dims(tf.convert_to_tensor(climate_snapshot[0][3], dtype=tf.float32), axis=-1),
            tf.expand_dims(tf.convert_to_tensor(climate_snapshot[1][3], dtype=tf.float32), axis=-1),
        ],
        axis=-1,
    )
    state.LR_snap = tf.concat(
        [
            tf.expand_dims(tf.convert_to_tensor(climate_snapshot[0][4], dtype=tf.float32), axis=-1),
            tf.expand_dims(tf.convert_to_tensor(climate_snapshot[1][4], dtype=tf.float32), axis=-1),
        ],
        axis=-1,
    )

def load_climate_data_one_snapshot(cfg, state, time):
    
    timestamp = int(time/100 + 20)  # Ensure the timestamp is the right format, and plus 20 to match the Trace21k dataset (in BC)

    # Temperature
    # Build filename by replacing the last number after '_' with the timestamp
    filename = cfg.processes.clim_chelsa_trace21k.temperature_filename + f"_{timestamp}.nc"

    ds = xr.open_dataset(filename)

    x = ds["x"].values.astype("float32").squeeze()
    y = ds["y"].values.astype("float32").squeeze()
    # flip y
    y = y[::-1]

    tasmax = ds["tasmax"].values.astype("float32").squeeze()
    tasmin = ds["tasmin"].values.astype("float32").squeeze()

    # convert to K not 10 K
    tasmax = tasmax / 10.0
    tasmin = tasmin / 10.0

    # Assume a sine curve between min and max, amplitude = (max - min) / 2, mean = (max + min) / 2
    # The standard deviation of a sine wave with amplitude A over one period is A / sqrt(2)
    amplitude = (tasmax - tasmin) / 2.0

    air_temp = (tasmax + tasmin)/ 2.0 - 273.15  # average of max and min temperature
    # air temperature offset
    air_temp += cfg.processes.clim_chelsa_trace21k.air_temp_offset
    air_temp_sd = amplitude / np.sqrt(2)  # Standard deviation of the sine wave, basically the root mean square of the sine wave
    air_temp = np.flip(air_temp, axis=1)  # flip axis 1
    air_temp_sd = np.flip(air_temp_sd, axis=1)  # flip

    # ========== # Precipitation
    filename = cfg.processes.clim_chelsa_trace21k.precipitation_filename + f"_{timestamp}.nc"
    ds = xr.open_dataset(filename)

    precipitation = ds["precipitation"].values.astype("float32").squeeze()
    # flip axis 1
    precipitation = np.flip(precipitation, axis=1)

    # surface elevation
    filename = cfg.processes.clim_chelsa_trace21k.dem_filename + f"_{timestamp}.nc"
    ds = xr.open_dataset(filename)  
    surf_clim_ref = ds["dem"].values.astype("float32").squeeze()
    # flip axis 0
    surf_clim_ref = np.flip(surf_clim_ref, axis=0)

    # replace -32768 with 0, which is the no data value in the CHELSA dataset
    surf_clim_ref = np.where(surf_clim_ref <= -32000, 0.0, surf_clim_ref)

    # vertical lapse rate
    filename = cfg.processes.clim_chelsa_trace21k.vertical_lapse_rate_filename + f"_{timestamp}.nc"
    ds = xr.open_dataset(filename)
    LR = ds["tz"].values.astype("float32")

    y_LR = ds["y"].values.astype("float32")
    x_LR = ds["x"].values.astype("float32")

    air_temp_sd = np.where(air_temp_sd > 0, air_temp_sd, 5.0)

    if not (y.shape == state.y.shape) & (x.shape == state.x.shape):
        # print("Resample climate data")

        air_tempN = np.zeros((air_temp.shape[0], state.y.shape[0], state.x.shape[0]))
        precipitationN = np.zeros(
            (air_temp.shape[0], state.y.shape[0], state.x.shape[0])
        )
        air_temp_sdN = np.zeros((air_temp.shape[0], state.y.shape[0], state.x.shape[0]))
        surf_clim_refN = np.zeros((state.y.shape[0], state.x.shape[0]))
        LRN = np.zeros((air_temp.shape[0], state.y.shape[0], state.x.shape[0]))

        for i in range(len(air_temp)):
            air_tempN[i] = RectBivariateSpline(y, x, air_temp[i])(state.y, state.x)
            air_temp_sdN[i] = RectBivariateSpline(y, x, air_temp_sd[i])(
                state.y, state.x
            )
            precipitationN[i] = RectBivariateSpline(y, x, precipitation[i])(
                state.y, state.x
            )
            
            # lapse rate
            # LRN[i] = RectBivariateSpline(y_LR, x_LR, LR[i])(state.y, state.x)
            # bivariate spline does not work for lapse rate with only one value in y direction, so big resolution
            if len(y_LR) > 4:
                LRN[i] = RectBivariateSpline(y_LR, x_LR, LR[i])(state.y, state.x)
            else:
                # calculate lapse rate indices of the middle of the domain
                y_mid = (state.y[0] + state.y[-1]) / 2.0
                x_mid = (state.x[0] + state.x[-1]) / 2.0
                # find the index of the closest value in y
                y_idx = np.argmin(np.abs(y_LR - y_mid))
                # find the index of the closest value in x
                x_idx = np.argmin(np.abs(x_LR - x_mid))
                # make matrix the same size as state.y and state.x
                LRN[i] = np.full((state.y.shape[0], state.x.shape[0]), LR[i][y_idx, x_idx])

        surf_clim_refN = RectBivariateSpline(y, x, surf_clim_ref)(state.y, state.x)

        air_temp = air_tempN
        air_temp_sd = air_temp_sdN
        precipitation = precipitationN
        surf_clim_ref = surf_clim_refN
        LR = LRN

    precipitation *= 12.0  # unit to [ kg * m^(-2) * month^(-1) ] -> [ kg * m^(-2) * y^(-1) ]
 
    return [air_temp, air_temp_sd, precipitation, surf_clim_ref, LR]


def interpolate_climate_over_a_year(climate_snapshot,resampling):
    """
    this applies interpolation, and shift to all climatic variables
    """
    climate_snapshot_out = []
    for cs in climate_snapshot:
        air_temp, air_temp_sd, precipitation, usurf, LR = cs

        # currently we do not resample in time, but keep monthly resolution to save memory
        # use 52 instead of 12 for weekly resolution, but this will increase memory usage
        air_temp = climate_upsampling_and_shift(air_temp, resampling=resampling)
        air_temp_sd = climate_upsampling_and_shift(air_temp_sd, resampling=resampling)
        precipitation = climate_upsampling_and_shift(precipitation, resampling=resampling)

        climate_snapshot_out.append([air_temp, air_temp_sd, precipitation, usurf, LR])

    return climate_snapshot_out

def climate_upsampling_and_shift(field, resampling, shift=0.0):
    """
    temporally resample (up) and shift to hydrological year
    """

    if resampling == len(field):
        Y = field

    else:
        assert resampling > len(field)
        mid = (field[-1] + field[0]) / 2.0
        x = np.concatenate(([0], (np.arange(len(field)) + 0.5) / len(field), [1]))
        y = np.concatenate(([mid], field, [mid]))
        X = (np.arange(resampling) + 0.5) / resampling
        #            Y = CubicSpline(x, y, bc_type='periodic', axis=0)(X)
        Y = interp1d(x, y, kind="linear", axis=0)(X)

    # the shift serves to adjust to hyroloogical year, i.e. start Oct 1st
    if shift > 0:
        Y = np.roll(Y, int(len(Y) * (1 - shift)), axis=0)

    return Y


def climate_donwsampling_and_shift(field, resampling, shift=0.75):
    """
    temporally resample (down) and shift to hydrological year
    """

    assert resampling < len(field)

    k = resampling
    m = field.shape[0]
    o = field.shape[1:]
    Y = np.nanmean(field[: (m // k) * k].reshape((m // k, k) + o), axis=0)
    Z = np.nanstd(field[: (m // k) * k].reshape((m // k, k) + o), axis=0)
    Z[Z < 0.1] = 0.1  # ensure the std is not zero

    # the shift serves to adjust to hyroloogical year, i.e. start Oct 1st
    if shift > 0:
        Y = np.roll(Y, int(len(Y) * (1 - shift)), axis=0)
        Z = np.roll(Z, int(len(Z) * (1 - shift)), axis=0)

    return Y, Z


def plot_first_read_results(cfg, state):
    """    plot the first read results
    """
    import matplotlib.pyplot as plt

    # plot dem, precipiation, air temperature, air temperature standard deviation, lapse rate
    fig, ax = plt.subplots(3, 2, figsize=(12, 12))
    ax[0, 0].imshow(state.tempsurfref, cmap="terrain", origin='lower' )
    # colorbar for surface elevation
    cbar = plt.colorbar(ax[0, 0].images[0], ax=ax[0, 0])
    cbar.set_label("Surface elevation [m a.s.l.]")

    # surface elevation of glacier model
    ax[0, 1].imshow(state.topg, cmap = "terrain", origin='lower')
    cbar = plt.colorbar(ax[0, 1].images[0], ax=ax[0, 1])
    cbar.set_label("Surface elevation of glacier model [m a.s.l.]")


    ax[2, 0].imshow(np.mean(state.precipitation,axis=0), cmap="Blues", origin='lower')
    # colorbar for precipitation
    cbar = plt.colorbar(ax[2, 0].images[0], ax=ax[2, 0])
    cbar.set_label("Precipitation [mm/month]")
    ax[1, 0].imshow(np.mean(state.air_temp,axis=0), cmap="coolwarm", origin='lower')
    # colorbar for air temperature
    cbar = plt.colorbar(ax[1, 0].images[0], ax=ax[1, 0])
    cbar.set_label("Air temperature [°C]")
    ax[1, 1].imshow(np.mean(state.air_temp_sd,axis=0), cmap="coolwarm", origin='lower')
    # colorbar for air temperature standard deviation
    cbar = plt.colorbar(ax[1, 1].images[0], ax=ax[1, 1])
    cbar.set_label("Air temperature standard deviation [°C]")
    ax[2, 1].imshow(np.mean(state.LR,axis=0), cmap="coolwarm", origin='lower')
    # colorbar for lapse rate
    cbar = plt.colorbar(ax[2, 1].images[0], ax=ax[2, 1])
    cbar.set_label("Lapse rate [K/km]") 

    # save plot
    plot_filename = os.path.join(state.original_cwd, f"climate_snapshot_{int(state.time_frames[0])}.png")
    plt.savefig(plot_filename)
    plt.close(fig)
