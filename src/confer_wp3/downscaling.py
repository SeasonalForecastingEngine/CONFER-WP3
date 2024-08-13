
import warnings
import numpy as np
import xarray as xr

from os import path
from functools import partial
from datetime import datetime
from calendar import monthrange
from scipy.interpolate import interp1d

from .utils import _preprocess, month_init_dict, get_filename, interpolate_forecasts


month_init_dict = {'jan':1 ,'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}


# Calculate the percentiles of the pentad climatology of the target data set

def calculate_target_percentiles(target, year_train_start, year_train_end, lon_bnds, lat_bnds, target_dir, filename_pct_target):
    requested_years = [*range(year_train_start, year_train_end+1)]
    filename_str = {'chirps':'chirps-v2.0', 'imerg':'imerg', 'rfe2':'rfe2'}[target]
    res_str = {'chirps':'_p25', 'imerg':'', 'rfe2':''}[target]
    available_years = [year for year in requested_years if path.exists(f'{target_dir}/{filename_str}.{year}.pentads{res_str}.nc')]
    if len(available_years) < len(requested_years):
        missing_years = ' '.join(map(str, list(set(requested_years).difference(set(available_years)))))
        warnings.warn(f"The following years of {target.upper()} data could not be loaded:"+"\n"+missing_years)
    if len(available_years) == 0:
        raise Exception("No files found for the selected training period.")
    file_list = [f'{target_dir}/{filename_str}.{year}.pentads{res_str}.nc' for year in available_years]
    partial_func = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
    print("Loading data ...")
    ds = xr.open_mfdataset(file_list, concat_dim='time', preprocess=partial_func, combine='nested', combine_attrs='drop_conflicts')
    pentad_length = np.append(ds.time.diff(dim='time').values.astype('timedelta64[D]').astype(float), 6)
    prcp_pentad_daily_avg = ds.precip.values / pentad_length[:,None,None]
    ndts, nlat, nlon = prcp_pentad_daily_avg.shape
    use_idx = np.any(np.isfinite(prcp_pentad_daily_avg), axis=(1,2))
    prcp_pentad_daily_avg = prcp_pentad_daily_avg[use_idx,:,:]
    pentad_of_year = np.arange(ndts)[use_idx] % 72
    prcp_pct = np.full((72,99,nlat,nlon), np.nan, dtype=np.float32)
    for ipt in range(72):
        if ipt % 10 == 0:
            print(f"Processing pentad {ipt+1}/72 ...")
        pentad_idx = np.minimum(abs(ipt-pentad_of_year),np.minimum(abs(ipt+72-pentad_of_year),abs(ipt-72-pentad_of_year))) <= 3
        prcp_pct[ipt,:,:,:] = np.percentile(prcp_pentad_daily_avg[pentad_idx,:,:], axis=0, q=range(1,100,1))
    print(f"Output saved as '{filename_pct_target}'.")
    da_prcp_pct = xr.DataArray(
        data= prcp_pct,
        dims=['pentad','level','lat','lon'],
        coords={'pentad': np.arange(1,73), 'level': np.arange(1,100), 'lat': ds.lat.values, 'lon': ds.lon.values,},
        name='percentile',
        attrs=dict(
            description=f'Climatological percentiles of {target.upper()} daily precipitation averages over a pentad',
            units='mm/day',),
        )
    da_prcp_pct.to_netcdf(filename_pct_target)



# Helper function to calculate daily average forecast precipitation amounts over each pentad

def calculate_pentad_daily_average(prcp_fcst, year_fcst, month_init, return_index=False):
    nmbs, nlts, nlatf, nlonf = prcp_fcst.shape
    nlm = nlts // 30
    npts = 6*nlm
   # Find indices that delineate the pentads
    pentad_end_idx = np.zeros(npts, dtype=np.int32)
    pentad_end_idx[0] = 4
    for imt in range(nlm):
        year_valid = year_fcst + (month_init_dict[month_init]+imt-1)//12
        month_valid = 1 + (month_init_dict[month_init]+imt-1)%12
        days_this_month = 0
        for ipt in range(5):
            days_this_month += 5
            if imt == 0 and ipt == 0:
                continue
            pentad_end_idx[6*imt+ipt] = pentad_end_idx[6*imt+ipt-1] + 5
        pentad_end_idx[6*imt+5] =  pentad_end_idx[6*imt+4] + monthrange(year_valid, month_valid)[1] - days_this_month
   # Calculate accumulations between these delineations
    prcp_fcst_pentad = np.full((nmbs,npts,nlatf,nlonf), np.nan, dtype=np.float32)
    prcp_fcst_pentad[:,0,:,:] = 200.*prcp_fcst[:,pentad_end_idx[0],:,:]
    for ipt in range(1,npts):
        ilt0 = pentad_end_idx[ipt-1]
        ilt1 = pentad_end_idx[ipt]
        prcp_fcst_pentad[:,ipt,:,:] = np.maximum(0.,1000.*(prcp_fcst[:,ilt1,:,:]-prcp_fcst[:,ilt0,:,:])/(ilt1-ilt0))
    if return_index:
        return prcp_fcst_pentad, pentad_end_idx
    else:
        return prcp_fcst_pentad



# Calculate the percentiles of the pentad climatology of the ensemble forecast data set

def calculate_forecast_percentiles(system, target, year_train_start, year_train_end, month_init, lon_target, lat_target, fcst_dir, filename_pct_fcst):
    requested_years = [*range(year_train_start, year_train_end+1)]
    available_years = [year for year in requested_years if path.exists(get_filename(fcst_dir, system, year, month_init))]
    
    if len(available_years) < len(requested_years):
        missing_years = ' '.join(map(str, list(set(requested_years).difference(set(available_years)))))
        warnings.warn(f"The following years of {system.upper()} forecast data could not be loaded:"+"\n"+missing_years)
    if len(available_years) == 0:
        raise Exception("No forecast data found to calculate percentiles.")
    file_list = [get_filename(fcst_dir, system, year, month_init) for year in available_years]
    lon_bnds = [np.floor(min(lon_target))-0.5, np.ceil(max(lon_target))+0.5]
    lat_bnds = [np.floor(min(lat_target))-0.5, np.ceil(max(lat_target))+0.5]
    partial_func = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
    nlon = len(lon_target)
    nlat = len(lat_target)
    nyrs = len(file_list)
    ds = xr.open_mfdataset(file_list[0], preprocess=partial_func)
    lon_fcst = ds.lon.values
    lat_fcst = ds.lat.values
    nmbs, nlts, nlatf, nlonf = ds.precip.shape
    ds.close()
    npts = 6 * (nlts//30)
    print("Loading data ...")
    prcp_pentad_daily_avg = np.full((nyrs,nmbs,npts,nlatf,nlonf), np.nan, dtype=np.float32)
    for iyr in range(nyrs):
        ds = xr.open_mfdataset(file_list[iyr], preprocess=partial_func)
        prcp_fcst = ds.precip.values[:nmbs,:nlts,:,:]
        ds.close()
        prcp_pentad_daily_avg[iyr,:,:,:,:] = calculate_pentad_daily_average(prcp_fcst, available_years[iyr], month_init)
    prcp_fcst_pct = np.full((npts,99,nlat,nlon), np.nan, dtype=np.float32)
    for ipt in range(npts):
        if ipt % 10 == 0:
            print(f"Processing pentad {ipt+1}/{npts} ...")
        prcp_pentad_daily_avg_ip = interpolate_forecasts(prcp_pentad_daily_avg[:,:,ipt,:,:], lat_fcst, lon_fcst, lat_target, lon_target)
        prcp_fcst_pct[ipt,:,:,:] = np.percentile(prcp_pentad_daily_avg_ip, axis=(0,1), q=range(1,100,1))
    print(f"Output saved as '{filename_pct_fcst}'.")
    da_prcp_pct = xr.DataArray(
        data= prcp_fcst_pct,
        dims=['pentad','level','lat','lon'],
        coords={'pentad': np.arange(1,npts+1), 'level': np.arange(1,100), 'lat': lat_target, 'lon': lon_target,},
        name='percentile',
        attrs=dict(
            description=f'Climatological percentiles of {system.upper()} daily precipitation averages over a pentad',
            units='mm/day',),
        )
    da_prcp_pct.to_netcdf(filename_pct_fcst)



# Helper function that performs the quantile mapping

def quantile_mapping(prcp_fcst, pctl_fcst, pctl_target):
    nmbs = len(prcp_fcst)
    x, ui = np.unique(np.round(np.append(0.0, pctl_fcst),2), return_index=True)
    y = np.append(0.0, pctl_target)[ui]
    if len(ui) == 1:
        prcp_fcst_bc = prcp_fcst.copy()          # no bias correction (at dry locations with almost all percentiles equal to zero)
    elif len(ui) == 2:
        slope1 = y[-1]/x[-1]
        prcp_fcst_bc = np.where(prcp_fcst<x[-1], slope1*prcp_fcst, y[-1]+min(slope1,1.5)*(prcp_fcst-x[-1]))    # linear extrapolation with reduced slope
    elif len(ui) == 3:
        prcp_fcst_bc = np.full(nmbs, np.nan, dtype=float)
        itp_ind1 = (prcp_fcst <= x[-2])
        itp_ind2 = np.logical_and(prcp_fcst > x[-2], prcp_fcst <= x[-1])
        extp_ind = (prcp_fcst > x[-1])
        slope1 = y[-2]/x[-2]
        slope2 = (y[-1]-y[-2])/(x[-1]-x[-2])
        prcp_fcst_bc[itp_ind1] = slope1*prcp_fcst[itp_ind1]                             # linear interpolation part 1
        prcp_fcst_bc[itp_ind2] = y[-2] + slope2*(prcp_fcst[itp_ind2]-x[-2])             # linear extrapolation part 2
        prcp_fcst_bc[extp_ind] = y[-1] + min(slope2,1.5)*(prcp_fcst[extp_ind]-x[-1])    # linear extrapolation with reduced slope
    else:
        prcp_fcst_bc = np.full(nmbs, np.nan, dtype=float)
        itp_ind1 = (prcp_fcst <= x[-3])
        itp_ind2 = np.logical_and(prcp_fcst > x[-3], prcp_fcst <= x[-1])
        extp_ind = (prcp_fcst > x[-1])
        slope1 = np.sum(y[-2:]-y[-3])/np.sum(x[-2:]-x[-3])
        itp_fct = interp1d(x, y, kind='linear', fill_value='extrapolate')
        prcp_fcst_bc[itp_ind1] = itp_fct(prcp_fcst[itp_ind1])                                                # linear interpolation below the 97th percentile
        prcp_fcst_bc[itp_ind2] = y[-3] + slope1*(prcp_fcst[itp_ind2]-x[-3])                                  # linear interpolation above 97th percentile, slope estimated with 98th/99th percentile
        prcp_fcst_bc[extp_ind] = y[-3] + slope1*(x[-1]-x[-3]) + min(slope1,1.5)*(prcp_fcst[extp_ind]-x[-1])  # linear extrapolation with reduced slope 
    return prcp_fcst_bc



# Load a new forecast and downscale to selected climatology via quantile mapping 

def downscale_forecasts(system, year_fcst, month_init, pctl_fcst, pctl_target, lon_target, lat_target, fcst_dir, filename_precip_dwnsc):
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
#    init_time = ds.time.values
    lon_fcst = ds.lon.values
    lat_fcst = ds.lat.values
    prcp_fcst = ds.precip.values
    ds.close()
    nmbs, nlts, nlatf, nlonf = prcp_fcst.shape
    prcp_daily = np.maximum(0., 1000.*np.diff(np.insert(prcp_fcst, 0, 0., axis=1), axis=1))
    prcp_daily_ip = interpolate_forecasts(prcp_daily, lat_fcst, lon_fcst, lat_target, lon_target)
    prcp_pentad_daily_avg, pentad_end_idx = calculate_pentad_daily_average(prcp_fcst, year_fcst, month_init, return_index=True)
    prcp_pentad_daily_avg_ip = interpolate_forecasts(prcp_pentad_daily_avg, lat_fcst, lon_fcst, lat_target, lon_target)
    prcp_daily_bc = np.full((nmbs,pentad_end_idx[-1]+1,nlat,nlon), np.nan, dtype=np.float32)
    npts = len(pentad_end_idx)
    for ipt in range(npts):                                   # ipt: pentad after forecast initialization
        jpt = (6*(month_init_dict[month_init]-1) + ipt) % 72  # jpt: pentad of year
        if ipt % 10 == 0:
            print(f"Processing pentad {ipt+1}/{npts} ...")
        if ipt == 0:
            iptb = 0
        else:
            iptb = pentad_end_idx[ipt-1] + 1
        ipte = pentad_end_idx[ipt] + 1
        for ix in range(nlon):
            for iy in range(nlat):
                if np.any(np.isnan(pctl_target[:,:,iy,ix])):
                    continue
                prcp_pentad_daily_avg_bc = quantile_mapping(prcp_pentad_daily_avg_ip[:,ipt,iy,ix], pctl_fcst[ipt,:,iy,ix], pctl_target[jpt,:,iy,ix])
                ratio = np.ones(nmbs, dtype=float)
                adj_ind = (prcp_pentad_daily_avg_ip[:,ipt,iy,ix] > 0.1)     # skip bias correction unless at least 0.1 mm average daily precipitation is predicted for this pentad
                ratio[adj_ind] = np.minimum(5., prcp_pentad_daily_avg_bc[adj_ind]/prcp_pentad_daily_avg_ip[adj_ind,ipt,iy,ix])
                prcp_daily_bc[:,iptb:ipte,iy,ix] = prcp_daily_ip[:,iptb:ipte,iy,ix] * ratio[:,None]
    print(f"Output saved as '{filename_precip_dwnsc}'.")
    da_prcp_daily_bc = xr.DataArray(
        data= prcp_daily_bc,
        dims=['ensemble','time','lat','lon'],
        coords={'ensemble': [*range(nmbs)], 'time': pd.date_range(f'{year_fcst}-{month_init_dict[month_init]:02}-01', periods=pentad_end_idx[-1]+1), 'lat': lat_target, 'lon': lon_target,},
        name='precip',
        attrs=dict(
            description=f'Downscaled {system.upper()} daily precipitation amounts',
            units='mm/day',),
        )
    da_prcp_daily_bc.to_netcdf(filename_precip_dwnsc)


