"""Isolate topg from output for new input"""

# classic imports
import netCDF4 as nc
import numpy as np
import os

# file path to the output file
output_file = "barkrak/output/2026-06-22/example/output.nc"
# load the output file
ds = nc.Dataset(output_file, "r")
# get the topg variable
topg = ds.variables["topg"][:]
# get the x, y coordinates
x = ds.variables["x"][:]
y = ds.variables["y"][:]
# get the time variable
time = ds.variables["time"][:]

# print the shape of the topg variable
print(f"Shape of topg: {topg.shape}")
print(f"Shape of x: {x.shape}")
print(f"Shape of y: {y.shape}")

# check if y differences are constant
dy = np.diff(y)
if np.allclose(dy, dy[0]):
    print("Y differences are constant.")
else:
    print("Y differences are not constant.")
    print(dy)

    # if the y are not constant, round them to the nearest integer, and make a new y array with constant spacing
    y_rounded = np.round(y)
    y_new = np.arange(y_rounded.min(), y_rounded.max() + 1, dy.max())
    print(f"New y array with constant spacing: {y_new}")

    # but y new needs to have the same number of points as the original,
    # so we need to interpolate the topg variable to the new y coordinates
    from scipy.interpolate import interp1d
    interp_func = interp1d(y, topg, axis=1, bounds_error=False, fill_value="extrapolate")
    topg_interpolated = interp_func(y_new)  # interpolate along the y axis
    topg = topg_interpolated

    y = y_new

# same for x
dx = np.diff(x)
if np.allclose(dx, dx[0]):
    print("X differences are constant.")
else:
    print("X differences are not constant.")
    print(dx)

    # if the x are not constant, round them to the nearest integer, and make a new x array with constant spacing
    x_rounded = np.round(x)
    x_new = np.arange(x_rounded.min(), x_rounded.max() + 1, dx.max())
    print(f"New x array with constant spacing: {x_new}")

    # but x new needs to have the same number of points as the original,
    # so we need to interpolate the topg variable to the new x coordinates
    interp_func = interp1d(x, topg, axis=2, bounds_error=False, fill_value="extrapolate")
    topg_interpolated = interp_func(x_new)  # interpolate along the x axis
    topg = topg_interpolated

    x = x_new

print(f"Shape of topg after interpolation: {topg.shape}")
print(f"shape of x and y: {x.shape}, {y.shape}")

# check if the topg variable has values above 8000
if np.any(topg > 8000):
    print("Topg variable has values above 8000.")
    print(f"Max value of topg: {topg.max()}")

    print("Setting values above 8000 to 0.0")
    topg[topg > 8000] = 0.0
topg[topg < -100.0] = 0.0

# create new netcdf with only last topg and thk frame
new_file = "barkrak/output/2026-06-22/example/barkrak_igm_topg.nc"
nc_out = nc.Dataset(new_file, "w", format="NETCDF4")
# create dimensions
nc_out.createDimension("x", len(x))
nc_out.createDimension("y", len(y))
# only take the last snapshot of topg and thk
# create variables
x_var = nc_out.createVariable("x", "f4", ("x",))
y_var = nc_out.createVariable("y", "f4", ("y",))
topg_var = nc_out.createVariable("topg", "f4", ("y", "x"))
# thk_var = nc_out.createVariable("thk", "f4", ("y", "x"))
# assign data to variables
x_var[:] = x
y_var[:] = y
topg_var[:, :] = topg[-1, :, :]  # only last time step
# thk_var[0, :, :] = ds.variables["thk"][-1, :, :]  # only last time step
# close the new netcdf file
nc_out.close()

# print min and max of topg
print(f"Min value of topg: {topg.min()}")
print(f"Max value of topg: {topg.max()}")