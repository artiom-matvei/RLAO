# DRL4PAPYRUS
DRL4AO (DRL4PAPYRUS) is a project consisting in adapting Jalo Nousiainen’s Policy Optimization for Adaptive Optics ([PO4AO](https://www.aanda.org/articles/aa/pdf/2022/08/aa43311-22.pdf)), a Model Based Reinforcement Learning algorithm that predicts Deformable Mirror (DM) voltage commands to be run with [OOPAO](https://github.com/cheritier/OOPAO), a Python Adaptive Optics simulation tool that enables us to simulate AO systems for Astronomical observations. In this project we mainly simulate the [PAPYRUS](https://hal.science/AO4ELT7/hal-04402883) run system at Observatoire Haute-Provence (OHP), using OOPAO.

## REQUIREMENTS
The code is written for Python 3 (version 3.8.8) and requires the following modules (OOPAO and PO4AO):
### OOPAO REQUIREMENTS

```
joblib        => paralleling computing
scikit-image  => 2D interpolations
astropy       => handling of fits files
pyFFTW        => optimization of the FFT  
mpmath        => arithmetic with arbitrary precision
jsonpickle    => json files encoding
aotools       => zernike modes and functionalities for atmosphere computation
numba         => required in aotools
numexpr       => optimized maths operations
psutil		  => access system info
mpmath		  => real and complex floating-point arithmetic with arbitrary precision
tqdm 		  => loading bar
```
If GPU computation is available:
```
cupy => GPU computation of the PWFS code (Not required)
```
### PO4AO & MISCELLANEOUS REQUIREMENTS
*Common modules with OOPAO not included
```
numpy       => numerical functions
matplotlib  => plot results
gym         => RL environment encapsulation
torch       => PO4AO / environment encapsulation for the integrator
tensorboard => PO4AO

```
## INSTALATION
The creation of a Python environment is always recommended.
For this project specifically we used [Anaconda](https://www.anaconda.com/) for the creation of both local and remote Python environments.
If you which to use [Anaconda](https://www.anaconda.com/) you should install it in your machine first and then run the following commands to create your environment:

```
conda create --name environment_name
```
```
conda activate environment_name
```
Install all the required modules (and sub dependencies) automatically:
```
conda install --yes --file requirements_conda.txt
                or with pip
pip install -r requirements_pip.txt                   
```
*You should be in the same folder as the requirements.txt file to be able to use the second command or pass it with the absolute path of the file.

### OOPAO VERSION

This project used the OOPAO version from 30/07/2024 last commit "Added SpatialFilter and update ShackHartmann and Telescope classes accordingly." from the main branch  

SHA code: 858f2daf970ee3d702d986617eeae5d4b3b52d51  

To download an specific version of OOPAO do the following:

1. git clone https://github.com/cheritier/OOPAO.git
2. git reset --hard SHA_code

There is no need to download and install OOPAO separately if you are cloning this project, as the files for OOPAO (ver 30/07/2024) are already included. You may though:
- Update the OOPAO version on your own
- Nest OOPAO as a [submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) in your copy of the project.

*We opted to not include OOPAO as an independent submodule within this git repository in order to avoid unexpected incompatibility problems with an OOPAO update during the development phase of the experiments.

## Running locally with VS code

When running the code locally with VS code if you open the project as a file, make sure to open it from the directory “drl4ao”, otherwise the OOPAO path won’t be found.

## SUBMITTING JOBS AND EXPERIMENT TAGS
In order to submit jobs within the cluster see the two bash files: [job_ao4elt_integrator.sh](drl4ao/job_ao4elt_integrator.sh) and [job_ao4elt_po4ao.sh](drl4ao/job_ao4elt_po4ao.sh). Running the code locally is also possible by directly executing the files [integrator_main.py](drl4ao/MAIN_CODE/integrator_main.py) and [mbrl_main.p](drl4ao/MAIN_CODE/mbrl_main.py). If you choose to run the experiments locally or through an interactive session, you might have to change the simulation parameters, experiment tags and output paths manually, directly in the code.

### EXPERIMENT TAG SYSTEM
Having a set of well- defined rules for labeling the experiment not only promotes  better project organization but also enables us to keep track of each result obtained and to automate the process of generating plots.

In this project we used the following system to identify experiment results:  

YYYYMMDD-HHMMSS_experiment type_experiment parameters_XXs  

Where:
- YYYYMMDD-HHMMSS is the date and time the experiment was launched
- experiment type identifies whether the experiment was run on an integrator or RL.
- experiment parameters distinguish which AO parameters were used for the simulation
- XXs determines the total simulated observation time in seconds (s)

* These tags are generated automatically upon running the code through jobs or the main files directly. You may adjust them before launching each simulation.

#### AO4ELT7 Experiment Tags
For AO4ELT7 experiments we had the following tags:
```
- WS10_r013: WindSpeed at 10m/s (average of all layers), r0 at 13cm 
- WS10_r08.6 WindSpeed at 10m/s (average of all layers), r0 at 0.86cm
- WS20_r013 WindSpeed at 20m/s (average of all layers), r0 at 13cm
- WS20_r08.6 WindSpeed at 20m/s (average of all layers), r0 at 0.86cm
```

So, two examples of AO4ELT7 experiment file names would be: 
```
- 20240819-183832_po4ao_WS10_r013_20s
- 20240819-183832_integrator_WS10_r013_20s
```

## PLOT FUNCTIONS
All the plot functions used to generate the images for the publications of AO4ELT7 and SPIE2024 are included in the [plots.py](drl4ao/MAIN_CODE/Plots/plots.py) file. This code was optimized taking in account the _experiment tag system_ explained above. To run the plots use the file [run_plots.py](drl4ao/MAIN_CODE/Plots/run_plots.py) that automatically looks for the experiment results in the given path, plots the selected plot and saves the images in the given output path.

## PUBLICATIONS
Two conference papers were published within the scope of this project:
- AO4ELT7 2023: [PAPYRUS at OHP: Predictive control with reinforcement
learning for improved performance](docs/camelo_AO4ELT_2023.pdf)
- SPIE 2024: [Reinforcement learning-based control law on PAPYRUS: simulations using different atmospheric conditions](docs/SPIE_2024_drl4ao.pdf)

*The code contained in this repository only corresponds to the experiments realized for AO4ELT7.

## ACKNOWLEDGEMENTS
DRL4AO was created mainly based on Jalo Nousiainen's [FitAO](https://github.com/jnousi/FitAO/tree/main) structure, using his [PO4AO](https://github.com/jnousi/PO4AO) MBRL code for the RL experiments. This project was  developed by Raissa Camelo hired under a _France Relance_ contract financed by CNRS. Any eventual publication using this code must acknowledge the CNRS.

## LICENSE
This is an internal project from LAM. Availability and sharing of the code must be discussed beforehand.
