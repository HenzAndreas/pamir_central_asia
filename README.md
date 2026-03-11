# Pamir Central Asia

*A repository to model glaciers in the Pamir, also trying to give access to other people.*

---

## 📦 Installation

### Requirements
- Conda/Miniconda
- Python 3.10.18

### Setup on Octopus (UNIL)
As of **10.03.2026**, the following steps worked on Octopus:

```bash
conda create -n igm-pamir
conda install python=3.10.18
```

Then, navigate to the folder containing `setup.py` and run:

```bash
pip install -e .
```

### Folder structure
pamir_central_asia/
├── README.md
├── setup.py
├── igm/
│   ├── all-igm-folders
│   ├── igm_run.py
├── dam-jailoo/ (one project at the moment)
│   ├── data/
│   │   ├── here-your-nc-file
│   │   └── and-ice-mask
│   ├── experiment/
│   │   ├── here are .yaml config files you want to use
│   ├── user/
│   │   ├── user-defined-modules (do not change unless you are familiar with it)
│   ├── output/
│   │   ├── your output
│   └── run_igm_run.py (if you need to run it in a debugger (like vs code python debugger) it can be useful to have a python file to run)

