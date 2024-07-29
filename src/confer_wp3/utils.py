
import warnings
import numpy as np
import xarray as xr

from os import path
from functools import partial

from scipy.interpolate import RectBivariateSpline


month_init_dict = {'jan':1 ,'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}


# Helper function for building the name of the forecast file

def get_filename(fcst_dir, system, year_fcst, month_init):
    if fcst_dir[-5:-1] == 'grib':
        return  f'{fcst_dir}{system}/{month_init}/{system}_{month_init}_{year_fcst}.grib'
    elif fcst_dir[-3:-1] == 'nc':
        return f'{fcst_dir}total_precipitation_{system}_{year_fcst}_{month_init_dict[month_init]}.nc'
    else:
        raise Exception("Unable to identify file type (grib/nc) based on file path")



# Helper function for loading data: renames coordinates, if necessary, and subsets to selected region

def _preprocess(x, lon_bnds, lat_bnds):
    coord_names = list(x.coords._names)
    coord_dict = {}
    for cstr in ['lat','lon']:#,'time']:
        idx_matched = [i for i in range(len(coord_names)) if coord_names[i].find(cstr)>=0]
        if len(idx_matched) != 1:
            raise Exception(f"Unable to identify '{cstr}' coordinate")
        if coord_names[idx_matched[0]] != cstr:
            coord_dict[coord_names[idx_matched[0]]] = cstr
        if list(x.keys())[0] != 'precip':
            coord_dict[list(x.keys())[0]] = 'precip'
    x = x.rename(coord_dict)
    if x.lat.values[0] < x.lat.values[-1]:
        return x.sel(lon=slice(*lon_bnds), lat=slice(*lat_bnds))
    else:
        return x.isel(lat=slice(None,None,-1)).sel(lon=slice(*lon_bnds), lat=slice(*lat_bnds))



# Helper function to bilinearly interpolate ensemble forecasts

def interpolate_forecasts(prcp_fcst, lat_fcst, lon_fcst, lat_target, lon_target):
    d1, d2, nlatf, nlonf = prcp_fcst.shape
    prcp_fcst_itp = np.full((d1,d2,len(lat_target),len(lon_target)), np.nan, dtype=np.float32)
    for i1 in range(d1):
        for i2 in range(d2):
            if np.all(np.isnan(prcp_fcst[i1,i2,:,:])):
                continue
            itpfct = RectBivariateSpline(lat_fcst, lon_fcst, prcp_fcst[i1,i2,:,:], kx=1, ky=1, s=0)
            prcp_fcst_itp[i1,i2,:,:] = itpfct.__call__(lat_target, lon_target, grid=True)
    return prcp_fcst_itp



# Load and interpolate raw ensemble forecast

def load_and_interpolate_forecast(system, year_fcst, month_init, lon_target, lat_target, fcst_dir):
    filename = get_filename(fcst_dir, system, year_fcst, month_init)
    if not path.exists(filename):
        raise Exception("No forecast data found for selected year {year_fcst}.")
    lon_bnds = [np.floor(min(lon_target))-0.5, np.ceil(max(lon_target))+0.5]
    lat_bnds = [np.floor(min(lat_target))-0.5, np.ceil(max(lat_target))+0.5]
    nlon = len(lon_target)
    nlat = len(lat_target)
    print("Loading and interpolating forecast data ...")
    partial_func = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
    ds = xr.open_mfdataset(filename, preprocess=partial_func)
    lon_fcst = ds.lon.values
    lat_fcst = ds.lat.values
    prcp_fcst = ds.precip.values
    ds.close()
    nmbs, nlts, nlatf, nlonf = prcp_fcst.shape
    prcp_daily = np.maximum(0., 1000.*np.diff(np.insert(prcp_fcst, 0, 0., axis=1), axis=1))
    prcp_daily_ip = interpolate_forecasts(prcp_daily, lat_fcst, lon_fcst, lat_target, lon_target)
    return prcp_daily_ip



