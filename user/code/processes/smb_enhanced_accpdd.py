#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file 

import tensorflow as tf

# from compute_gradient_tf import compute_gradient_tf
from igm.utils.grad.grad import grad_xy

def initialize(cfg, state):
    state.tlast_mb = tf.Variable(-1.0e5000)


# Warning: The decorator permits to take full benefit from efficient TensorFlow operation (especially on GPU)
# Note that tf.function works best with TensorFlow ops; NumPy and Python calls are converted to constants.
# Therefore: you must make sure any variables are TensorFlow Tensor (and not Numpy)
# @tf.function()
def update(cfg, state):
    """
    mass balance forced by climate with accumulation and temperature-index melt model
    Input:  state.precipitation [Unit: kg * m^(-2) * y^(-1) water eq]
            state.air_temp      [Unit: °C           ]
    Output  state.smb           [Unit: m ice eq. / y]

    This mass balance routine implements a combined accumulation / temperature-index model [Hock, 2003].
    It is a TensorFlow re-implementation similar to the one used in the aletsch-1880-2100 example
    but adapted to fit as closely as possible (thought it is not a strict fit)
    the Positive Degree Day model implemented in PyPDD (Seguinot, 2019) used for the Parralel Ice Sheet
    Model (PISM, Khroulev and the PISM Authors, 2020) necessary to perform PISM / IGM comparison.
    The computation of the PDD using the expectation integration formulation (Calov and Greve, 2005),
    the computation of the snowpack, and the refereezing parameters are taken from PyPDD / PISM implementation.

    References:

    Hock R. (2003). Temperature index melt modelling in mountain areas, J. Hydrol.

    Seguinot J. (2019). PyPDD: a positive degree day model for glacier surface mass balance (v0.3.1).
    Zenodo. https://doi.org/10.5281/zenodo.3467639

    Khroulev C. and the PISM Authors. PISM, a Parallel Ice Sheet Model v1.2: User’s Manual. 2020.
    www.pism-docs.org

    Calov and Greve (2005), A semi-analytical solution for the positive degree-day model with
    stochastic temperature variations, JOG.
    """

    # update smb each X years
    if (state.t - state.tlast_mb) >= cfg.processes.smb_enhanced_accpdd.update_freq:
        if hasattr(state, "logger"):
            state.logger.info(
                "Construct mass balance at time : " + str(state.t.numpy())
            )

        # keep solid precipitation when temperature < smb_enhanced_accpdd_thr_temp_snow
        # with linear transition to 0 between smb_enhanced_accpdd_thr_temp_snow and smb_enhanced_accpdd_thr_temp_rain
        accumulation = tf.where(
            state.air_temp <= cfg.processes.smb_enhanced_accpdd.thr_temp_snow,
            state.precipitation * cfg.processes.smb_enhanced_accpdd.precip_offset,
            tf.where(
                state.air_temp >= cfg.processes.smb_enhanced_accpdd.thr_temp_rain,
                0.0,
                state.precipitation
                * (cfg.processes.smb_enhanced_accpdd.thr_temp_rain - state.air_temp)
                / (cfg.processes.smb_enhanced_accpdd.thr_temp_rain - cfg.processes.smb_enhanced_accpdd.thr_temp_snow),
            ),
        )

        # accumulation done, now compute the positive degree day (PDD) for the melt
        if cfg.processes.smb_enhanced_accpdd.type_of_aspect_correction == "temperature":
            # if aspect correction is used for temperature, we need to compute the incidence angle of the sun on the surface, and correct the temperature accordingly
            solar_elevation = 60.0 # degrees above the horizon
            solar_azimuth = 180.0 # degrees from north, clockwise
            incidence_angle = calc_incidence_angle(state.usurf, state.dX, solar_elevation, solar_azimuth)
            # correct temperature by a factor a factor reducing or upgrading by 100 %
            # depending on the incidence angle
            # 0.0 --> -100%
            # 60.0 --> 0%
            # 120.0 --> 100%
            aspect_temperature_correction = incidence_angle / 30.0 - 1.0 # gives a values between -1 and 1
            aspect_temperature_correction *= cfg.processes.smb_enhanced_accpdd.aspect_scaling_factor # scale the correction, this is a factor to be determined, if zero, no aspect correction is applied.
        else:
            aspect_temperature_correction = 0.0

        air_temp = state.air_temp - aspect_temperature_correction # minus sign means we have to make it cooler, where incidence angles are bigger --> north side

        if hasattr(state, "air_temp_sd"):
            # compute the positive temp with the integral formaulation from Calov and Greve (2005)
            # the formulation assumes the air temperature follows a normal distribution, it is obtained
            # by integrating by part of the integral of T * normal density over {T>=0} since
            # only positive temp. matters. This yields to a first boundary terms, and a second
            # involving the integral of the normal density, i.e. the erf function.
            cela = air_temp / (1.4142135623730951 * state.air_temp_sd)
            pos_temp_year = (
                state.air_temp_sd
                * tf.math.exp(-tf.math.square(cela))
                / 2.5066282746310002
                + air_temp * tf.math.erfc(-cela) / 2.0
            )
        else:
            pos_temp_year = tf.where(air_temp > 0.0, air_temp, 0.0)

        # unit to [  kg * m^(-2) * y^(-1) water eq  ] -> [ m water eq ]
        accumulation /= (accumulation.shape[0] * cfg.processes.smb_enhanced_accpdd.wat_density) 

        # unit to [ °C ]  -> [ °C y ]
        pos_temp_year /= pos_temp_year.shape[0]  

        ablation = []  # [ unit : water-eq m ]

        snow_depth = tf.zeros((state.air_temp.shape[1], state.air_temp.shape[2]))

        for kk in range(state.air_temp.shape[0]):
            # shift to hydro year, i.e. start Oct. 1
            k = (
                kk + int(state.air_temp.shape[0] * cfg.processes.smb_enhanced_accpdd.shift_hydro_year)
            ) % (state.air_temp.shape[0])

            # add accumulation to the snow depth
            snow_depth += accumulation[k]

            # the ablation (unit is m water eq.) is the product of positive temp  with melt
            # factors for ice, or snow, or a fraction of the two if all snow has melted
            ablation.append(
                tf.where(
                    snow_depth == 0,
                    pos_temp_year[k] * cfg.processes.smb_enhanced_accpdd.melt_factor_ice,
                    tf.where(
                        pos_temp_year[k] * cfg.processes.smb_enhanced_accpdd.melt_factor_snow
                        < snow_depth,
                        pos_temp_year[k] * cfg.processes.smb_enhanced_accpdd.melt_factor_snow,
                        snow_depth
                        + (
                            pos_temp_year[k]
                            - snow_depth / cfg.processes.smb_enhanced_accpdd.melt_factor_snow
                        )
                        * cfg.processes.smb_enhanced_accpdd.melt_factor_ice,
                    ),
                )
            )

            # remove snow melt to snow depth, and cap it as snow_depth can not be negative
            snow_depth = tf.clip_by_value(snow_depth - ablation[-1], 0.0, 1.0e10)

        ablation = (1 - cfg.processes.smb_enhanced_accpdd.refreeze_factor) * tf.stack(ablation, axis=0)

        # sum accumulation and ablation over the year, and conversion to ice equivalent
        state.smb = tf.math.reduce_sum(accumulation - ablation, axis=0)* (
            cfg.processes.smb_enhanced_accpdd.wat_density / cfg.processes.smb_enhanced_accpdd.ice_density
        )

        if cfg.processes.smb_enhanced_accpdd.type_of_aspect_correction == "smb":

            # if icemask is present, set smb to -10 where icemask > 0.5 or smb < 0        
            # # ============= Andreas Henz - Aspect corection =============
            # if cfg.processes.smb_enhanced_accpdd.aspect_correction: from August 2025 aspect correction is always applied and scaled over the scaling factor, if zero (default), no aspect correction is made.
            # aspect_scaling_factor = 1.0
            solar_elevation = 60.0 # degrees above the horizon
            # surface, dx, solar_elevation, solar_azimuth=180.0
            incidence_angle = calc_incidence_angle(state.usurf, state.dX, solar_elevation, solar_azimuth=180.0)
            
            # correct smb by a factor a factor reducing or upgrading by 100 %
            # depending on the incidence angle
            # 0.0 --> -100%
            # 30.0 --> 0%
            # 60.0 --> 100%
            
            aspect_correction = incidence_angle / 30.0 - 1.0 # gives a values between -1 and 1 [m]
            aspect_correction *= cfg.processes.smb_enhanced_accpdd.aspect_scaling_factor # scale the correction, this is a factor to be determined, if zero, no aspect correction is applied.

            state.smb += aspect_correction
        
        # =========== finished ===================        

        if hasattr(state, "icemask"):
            state.smb = tf.where(
                (state.icemask > 0.5), state.smb, -10
            )

        state.smb = tf.clip_by_value(state.smb, -100, cfg.processes.smb_enhanced_accpdd.smb_maximum_accumulation)

        state.tlast_mb.assign(state.t)


def finalize(cfg, state):
    pass


def calculate_normal_vector(elevation_array, dX):
    # Calculate gradients (slope in x and y directions)
    # !!!!!!!!!!!! This definition is very critical !!!!!!!!!!!!!
    # dzdx, dzdy = compute_gradient_tf(elevation_array, dx ,dx)  # this is very critical!!!
    dzdx, dzdy = grad_xy(elevation_array, dX, dX, staggered_grid=False, mode='extrapolate') # this is very critical what we define
    
    # Create the normal vector components
    normal_vector = tf.stack([-dzdx, -dzdy, tf.ones_like(dzdx)], axis=-1)

    # Normalize the normal vector
    norm = tf.norm(normal_vector, axis=-1, keepdims=True)
    normalized_normal_vector = normal_vector / norm

    return normalized_normal_vector

def calculate_sun_vector(elevation_angle, azimuth_angle):
    # Convert angles from degrees to radians
    elevation_rad = tf_deg2rad(elevation_angle)
    azimuth_rad = tf_deg2rad(azimuth_angle)
    
    # Calculate the components of the sun vector
    sun_vector = tf.stack([
        tf.cos(elevation_rad) * tf.sin(azimuth_rad),  # x-component
        tf.cos(elevation_rad) * tf.cos(azimuth_rad),  # y-component
        tf.sin(elevation_rad)                         # z-component
    ])
    
    # Normalize the sun vector
    norm = tf.norm(sun_vector)
    normalized_sun_vector = sun_vector / norm
    
    return normalized_sun_vector

def calc_incidence_angle(surface, dx, solar_elevation, solar_azimuth=180.0):
    # The angle of the sun vector to the surface normal vector is the cosine of the dot product
    sun_vector = calculate_sun_vector(solar_elevation, solar_azimuth)
    normal_vectors = calculate_normal_vector(surface, dx)
    
    # Calculate the dot product
    dot_product = tf.reduce_sum(normal_vectors * sun_vector, axis=-1)  # Efficiently computes the dot product for each normal vector
    
    # Clip the dot product to avoid domain errors in arccos
    dot_product = tf.clip_by_value(dot_product, -1.0, 1.0)
    
    # Calculate the angle in radians and then convert to degrees
    incidence_angles_rad = tf.acos(dot_product)
    incidence_angles_deg = tf_rad2deg(incidence_angles_rad)
    
    return incidence_angles_deg

def tf_rad2deg(rad):
    pi_on_180 = 0.017453292519943295
    return rad / pi_on_180

def tf_deg2rad(deg):
    pi_on_180 = 0.017453292519943295
    return deg * pi_on_180
