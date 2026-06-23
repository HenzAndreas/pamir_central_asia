import sys
import os

from igm.igm_run import main

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Running igm_run with parameters: {sys.argv[1:]} at {os.path.basename(__file__)}")

# define here your params to overwrite the default ones
list_params = [
    "+experiment=example"
# - processes.clim_chelsa_trace21k.air_temp_offset=-3.0
# - processes.smb_enhanced_accpdd.precip_offset=2.5
]


# elaborate print statement with time stamp and list of parameters
# print(f"Running igm_run with parameters: {additional_params} at {np.datetime64('now', 's')}")

# Change working directory to the directory containing the config files
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Simulate: python igm_run.py +experiment=params_simple
sys.argv = ["igm_run.py"] + list_params

main()