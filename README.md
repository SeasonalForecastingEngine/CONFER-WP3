# CONFER-WP3
This repository contains **Python code** developed within several sub-projects of work package 3 of the [CONFER](https://confer-h2020.eu/) project, one of the European Union’s Horizon 2020 research and innovation programmes.

An **R-package for forecast evaluation** was also developed in this work package, and can be found [here](https://github.com/SeasonalForecastingEngine/SeaVal).



## Installation

To install the package, run
* `pip install git+ssh://git@github.com:SeasonalForecastingEngine/CONFER-WP3.git`



## Probabilistic prediction of rainy season onset dates at seasonal forecast lead times

In this sub-project, code was developed to generate probabilistic prediction of rainy season onset dates based on seasonal forecasts of daily precipitation amounts, following the research documented in this [journal article](https://link.springer.com/article/10.1007/s00382-023-07085-y).

The routines required to obtain ensemble of rany season onset dates can be driven from the Jupyter notebook 'onset_example.ipynb'. They require seasonal forecast and CHIRPS data (see section **Required data sets** below for more details) that is assumed to be available in directories that can be specified at the beginning of the notebook.



## Statistical downscaling of seasonal forecasts of daily precipitation amounts

In this sub-project, code was developed to statistically bias-correct and downscale seasonal forecasts of daily precipitation amounts so that they can be used e.g. as inputs to hydrological prediction systems. Depending on the region, either CHIRPS or IMERG precipitation data is considered to be the most suitable proxy for observed precipitation, so our algorithm is set up such that either of the two data sets can be used. While the final output is generated at daily temporal resolution, the bias-correction is perfomed at the pentad aggregation level, which saves computation time and memory, and puts more emphasis on the correct representation of multi-day precipitation accumulations.

The downscaling routines and necessary data pre-processing steps can be driven from the Jupyter notebook 'downscaling_example.ipynb'. They assume that the required data (see section **Required data sets** below for more details) have been already downloaded to directories that can be specified at the beginning of the notebook. The directories where the output (i.e., the downscaled forecasts) and the intermediate outputs (percentiles of model and observation climatology) should be stored can also be specified in the notebook.



## Machine learning based probabilistic prediction of seasonal precipitation amounts

In this sub-project, code was developed to predict the probabilities of below/above normal seasonal precipitation amounts based on a range of climate indices. All associated rountines can be driven from the Jupyter notebook 'lasso_fcst_example.ipynb'. The prediction target is constructed from CHIRPS precipitation data, the indices are derived from ERA5 reanalysis data (see section **Required data sets** below for more details) which are assumed to be available in directories that can be specified at the beginning of the notebook. The directories where the output (tercile probability forecasts) and intermediate outputs (predictor and predictand anomalies, EOFs, etc.) should be stored can also be specified in the notebook.



## Required data sets

### CHIRPS v2.0 precipitation data

All three sub-projects use CHIRPS v2.0 precipitation data at different aggregation levels.

For rainy season onset prediction, daily precipitation accumulations at 0.25 degree horizonal resolution are used directly in the form that can be downloaded [here](https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p25/). The scripts associated with rainy season onset prediction described above assumes that exact structure of filenames and variable names within each file.

The downscaling of seasonal forecasts for use in streamflow prediction is performed at the pentad aggregation level, for which CHIRPS data can be downloaded [here](https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_pentad/netcdf/) at 0.05 degree horizonal resolution. This hight resolution, however, poses some challenges with memory when the downscaling is performed over a larger region, and we therefore upscaled the data to 0.25 degree horizonal resolution. A Python script 'upscale_chirps_pentads.py' that performs this task is included in this package in the 'src/confer_wp3' folder and can be run from the command line with the path to the CHIRPS data as an argument, e.g.:

`python upscale_chirps_pentads.py /data/my_chirps_directory/`

For every original file 'chirps-v2.0.<year\>.pentads.nc', the script will create a file 'chirps-v2.0.<year\>.pentads_p25.nc' in the same directory and with the same variable names, and these naming conventions are assumed by the scripts associated with statistical downscaling of seasonal forecasts described above.

The machine learning based probabilistic prediction of seasonal precipitation amounts uses CHIRPS data at the monthly aggregation level, which can be downloaded [here](https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/) as a single file.


### IMERG precipitation data

For the downscaling of seasonal forecasts for use in streamflow prediction we alternatively facilitate the use of IMERG precipitation data which can e.g. be downloaded [here](https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHH_06/summary?keywords=%22IMERG%20final%22) at 0.1 degree horizonal resolution. Regardless of the data source, it is assumed that the data are aggregated to pentads and stored as a set of files of the type 'imerg.<year\>.pentads.nc' analoguous to the CHIRPS data described above. The 0.1 degree horizonal resolution did not pose any problems for the domain considered in this project and is assumed by the scripts associated with the downscaling of seasonal forecasts.


### ERA5 reanalysis data

ERA5 reanalysis data can be downloaded from the [Climate Data Store (CDS)](https://cds.climate.copernicus.eu/#!/home). *Add more detail about the assumed file structure in which the files are stores*.


### Seasonal forecasts of daily precipitation amounts

Seasonal forecasts of daily precipitation amounts by various forecast centers can also be downloaded from the [Climate Data Store (CDS)](https://cds.climate.copernicus.eu/#!/home). Our code for reading the forecasts assumes one of the following filename and format conventions:
* If the name of the folder in which the forecasts are stored ends in '_nc', all filenames are assumed to have the form 'total_precipitation\_<forecast system\>\_<forcast year\>\_<initialization month\>.nc' with the choices of a forecast system, training period, and forecast initialization month specified in the respective Jupyter notebooks.
* If the name of the folder in which the forecasts are stored ends in '_grib`', all filenames are assumed to have the form '<forecast system\>\_<initialization month\>\_<forcast year\>.grib' with the choices of a forecast system, training period, and forecast initialization month specified in the respective Jupyter notebooks.

