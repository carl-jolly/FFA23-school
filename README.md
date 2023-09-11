# FFA23-school
OPAL input files and python scripts for FFA23 school  

See the install instructions pdf for installing OPAL and setting up a python environment  

The OPAL tutorial pdf and jupyter notebook explain how to setup, run and analyse an OPAL simulation  

Dependancies:  
OPAL-2022.1 or greater  
Python packages:  
Numpy, Scipy, Matplotlib, Pandas  
Also PyNAFF (optional)  

Files:  
OPAL input files – DF_lattice, FFA_FODO_lattice  
OPAL input distribution files – dist.dat, CO_coords_3MeV.dat, perturbed_coords_3MeV.dat  

Python files:  
FFA23_OPAL_tutorial.ipynb – Tutorial jupyter notebook.  

plot_dat.py – Plotting functions for OPAL output files.  
optimiseCO.py – Closed orbit finder for FFAs in OPAL.  
read_dat.py – Reads OPAL output files into pandas DataFrames.  
probes_tune_calc.py - Calculates the cell tune using the OPAL probe.loss output file.
