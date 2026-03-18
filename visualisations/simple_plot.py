
# imports
import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset # important to read netCDF files, if you do not have it, you can install it via pip install netCDF4
import os
os.getcwd()

# get color from cmeri
import cmcrameri.cm as cmc # if that import does not work, you can either install it or use other colormaps from matplotlib (e.g., plt.cm.viridis)
import matplotlib.colors as colors

# load dataset
output_dir = "dam-jailoo/output/2026-03-11/example/"

ds = Dataset(output_dir+"output.nc")
print('Available variables in the netCDF file:')
for var in ds.variables:
    print(f' - {var}: {ds.variables[var].shape}')

# get topg
topg = ds.variables['topg'][0,:,:]
thk = ds.variables['thk'][:, : ,:]
smb = ds.variables['smb'][:,:,:]
# other variables if you want....
# .....
# 
time_bp = -ds.variables['time'][:] # time as Before Present (BP)
x = ds.variables['x'][:]/1000 # convert to km
y = ds.variables['y'][:]/1000 # convert to km
dx = x[1]-x[0]

# make an easy plot of the topography and ice thickness
fig, ax = plt.subplots(figsize=(8,6))
ax.imshow(topg, cmap='Greys_r', origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])

# add ice thickness of the last time step
thk_last = thk[-1,:,:]
time_last = time_bp[-1]

# plot ice thickness as a transparent layer on top of the topography
thk_masked = np.ma.masked_where(thk_last <= 0, thk_last) # mask out non-ice areas
cmap = cmc.devon # replacable with Blues, if you do not have cmcrameri
norm = colors.Normalize(vmin=0, vmax=np.max(thk_last))
ax.imshow(thk_masked, cmap=cmap, norm=norm, alpha=0.7, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])

# add text box with time information
textstr = f'Time: {time_last:.1f} years BP'
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

# add colorbar for ice thickness
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Ice Thickness (m)')

ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')

# save figure here
plt.savefig("topography_ice_thickness.png", dpi=300)