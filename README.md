# ContinentalLithopshereThermalModel
This repository contains a Python-based Continental Lithosphere Thermal Model. This model simulates the best-fit 1D steady-state conductive thermal profiles locally and integrates into a 3D model. This model also includes components for deriving the seismogenic thicknes, and a numerically solved 2D heat diffusion model.

Here contains the Python function 'ContinentalThermalModel.py', configuration text file 'ContinentalThermalModel_config.txt', and model documentation and instruction Jupyter source file 'ThermalModel_Documentation.ipynb'. 

Input temperature proxies for the western US preferred model are also located in the folder 'Temperature_proxies', including surface heat flow (Mordensky et al., 2023) 'heat_flow.tif', crustal thickness (Buehler and Shearer, 2017) 'Moho_depth.tif', and Moho temperature 'Moho_T.tif' and its uncertainty 'Moho_T_Uncer.tif' datasets (Schutt et al., 2018).

Western US earthquake catalog 'EQ_catalog_WUS.csv' for seismogenic thickness calculation can be accessed at our Zenodo repository (https://zenodo.org/records/15390785).
