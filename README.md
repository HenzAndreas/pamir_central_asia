# Pamir Central Asia

*A repository to model glaciers in the Pamir, also trying to give access to other people.*

## IGM
*The Instructed Glacier Model (IGM) is coming from this repository: https://github.com/instructed-glacier-model/igm
Please always cite and refer to this page, when you are using the model*

---

## 📦 Installation

### Requirements
- Conda/Miniconda
- Python 3.10.18

### Setup 
You need to have conda installed, best (for windows user) is a WSL environment with Ubuntu installed. This works quite well: https://learn.microsoft.com/en-us/windows/wsl/install

Then have Miniconda (or any other conda distribution) installed: https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer

If Conda is running (in the base environment):

As of **10.03.2026**, the following steps worked on Octopus (Octopus Cluster, UNIL):

```bash
conda create -n igm-pamir
```
```bash
conda activate igm-pamir
```
```bash
conda install python=3.10.18
```

Then, navigate to the folder containing `setup.py` and run:

```bash
pip install -e .
```

### 📁 Folder Structure

```bash
pamir_central_asia/
├── README.md
├── setup.py
├── igm/
│   ├── all-igm-folders/
│   └── igm_run.py
├── data/                # here some basic climate data for example
│   └── trace21k-pamir
│   │   ├── lapse_rate_monthly
│   │   ├── precipitation_monthly
│   │   ├── temperature_monthly
│   │   └── dem
├── visualisations/       # here some simple scripts to visualise the netCDF output files
│   ├── simple_plot.py
│   └── simple_animation.py
├── dam-jailoo/          # One project at the moment
│   ├── data/
│   │   ├── here-your-nc-input-file # topography, initial ice thicknes, etc.
│   │   └── and-ice-mask # as a shapefile (e.g., exported from GIS software in the right coordinate system)
│   ├── experiment/
│   │   └── here-are-.yaml-config-files-you-want-to-use
│   ├── user/
│   │   └── user-defined-modules  # Do not change unless you are familiar with it
│   ├── output/
│   │   └── your-output
│   └── run_igm_run.py   # Useful for running in a debugger (e.g., VS Code)
```

### 📎 netCDF file handling

To look at netCDF files very quickly you can use the ncview tool in the WSL or Linux environment, which is installed with the following command:

```bash
sudo apt install ncview
```

Then you can open a netCDF file with:

```bash
ncview your_file.nc
```

Remember this works only on WSL or Linux, not on Windows directly. And it also does not look super nice, but gives you a fast impression of the data. For a detailed analysis, you should use Python script that load the netCDF files. I have placed a simple example for this in `visualisations/simple_plot.py`, which you can adapt to your needs. You can also have a look at the igm-internal plotting modules like [live dashboard](https://igm-model.org/latest/modules/outputs/live_dashboard/) or [plot2d](https://igm-model.org/latest/modules/outputs/plot2d/).

You can also use more advanced tools like Panoply (https://www.giss.nasa.gov/tools/panoply/) or Python libraries like xarray, which are more powerful and flexible for data analysis and visualisation.

#### compressing netCDF files
Sometimes the files are big, you can compress them with the following command (after installing nco tools: `sudo apt install nco`):
Actually it is anyways recommended, because ncview and also python libraries can read compressed netCDF files, 

```bash
ncks -L 1 variable_name input_file.nc output_file.nc
```
I recommend to compress only to level 1 (-L 1), because higher levels take much longer to compress but are not much smaller size. However, compressing reduces the files size by about a factor of 2, when there are many empty (zero) values, which is the case for the ice thickness files, where most of the grid points are zero (no ice).

#### extracting variables
If you want to extract only a specific variable from a netCDF file, you can use the
```bash
ncks -O -v variable_name input_file.nc output_file.nc
```
command. This is useful if you want to look at a specific variable, but the file contains many variables that you do not need. It also reduces the file size, which can be helpful for visualization or sharing.