"This is a script to plot the loaded topography and switch the y-axis if needed to make the nc file work"
"Andreas Henz - 22.06.2026"

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

# Load the topography data from the nc file
nc_file = 'barkrak/data/Barkrak_igm.nc'  # Replace with your nc file 

# load nc file
dataset = nc.Dataset(nc_file)
# load x, y, and topg
x = dataset.variables['x'][:]
y = dataset.variables['y'][:]
topg = dataset.variables['topg'][:]
# print other variables available
print(dataset.variables.keys())

# define vmin vmax in a reasonable way to make the plot look good
vmin = np.nanmin(topg)
vmax = np.nanmax(5000)

# Plot the topography
plt.figure(figsize=(10, 6))
plt.imshow(topg, extent=[x.min(), x.max(), y.min(), y.max()], cmap='terrain', origin='lower',
           vmin=vmin, vmax=vmax)
# Add colorbar and labels
plt.colorbar(label='Topography (m)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.savefig(f"test_topography_of_file_{nc_file.split('/')[-1].split('.')[0]}.png")
plt.show()

# if wanted
flip = True
if flip:
    # flip the y-axis, only topg and save as a new nc file
    topg_flipped = np.flip(topg, axis=0)
    y_flipped = np.flip(y, axis=0)
    # save as new nc file
    new_nc_file = f"flipped_{nc_file.split('/')[-1]}"
    with nc.Dataset(new_nc_file, 'w', format='NETCDF4') as new_dataset:
        # create dimensions
        new_dataset.createDimension('x', len(x))
        new_dataset.createDimension('y', len(y))
        # create variables
        x_var = new_dataset.createVariable('x', 'f4', ('x',))
        y_var = new_dataset.createVariable('y', 'f4', ('y',))
        topg_var = new_dataset.createVariable('topg', 'f4', ('y', 'x'))
        # assign data to variables
        x_var[:] = x
        y_var[:] = y_flipped
        topg_var[:, :] = topg_flipped
    print(f"Flipped topography saved to {new_nc_file}")