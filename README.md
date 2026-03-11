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

### Setup (worked like this on Octopus, UNIL)
As of **10.03.2026**, the following steps worked on Octopus:

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
├── dam-jailoo/          # One project at the moment
│   ├── data/
│   │   ├── here-your-nc-file
│   │   └── and-ice-mask
│   ├── experiment/
│   │   └── here-are-.yaml-config-files-you-want-to-use
│   ├── user/
│   │   └── user-defined-modules  # Do not change unless you are familiar with it
│   ├── output/
│   │   └── your-output
│   └── run_igm_run.py   # Useful for running in a debugger (e.g., VS Code)
```
