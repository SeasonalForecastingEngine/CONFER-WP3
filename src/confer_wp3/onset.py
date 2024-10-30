import warnings
import numpy as np
import xarray as xr

from os import path
from functools import partial
from datetime import datetime, timedelta

from scipy.interpolate import interp1d

from .glp import domain_boundaries, global_parameters
from .utils import _preprocess, month_init_dict, get_filename, interpolate_forecasts



# Helper function that identifies the rainy season onset day for given threshold exceedances and length of critical dry period

def find_onset_day(exc1d, exc3d, wnd_dry_spell, len_dry_spell):
    n = len(exc1d)
    wet_spell = np.logical_and(exc3d[:(n-2)], exc1d[:(n-2)])
    ind_dry_spell = np.expand_dims(np.arange(len_dry_spell),0)+np.expand_dims(np.arange(n-len_dry_spell+1),0).T
    dry_spell = np.all(~exc1d[ind_dry_spell], axis=1)
    ind_dry_spell_wnd = np.expand_dims(np.arange(wnd_dry_spell+1-len_dry_spell),0)+np.expand_dims(np.arange(n-2-wnd_dry_spell),0).T
    onset = np.logical_and(wet_spell[:(n-2-wnd_dry_spell)], ~np.any(dry_spell[3:][ind_dry_spell_wnd], axis=1))
    onset_day = -1 if np.all(~onset) else np.nonzero(onset)[0][0]+1
    return onset_day



# Function that calculates historical onset dates from CHIRPS data

def calculate_onset_hist(region, month_start, year_clm_start, year_clm_end, thr_dry, thr_wet, wnd_dry_spell, len_dry_spell, chirps_dir):
    lon_bnds, lat_bnds = domain_boundaries(region)
    day_start, nwks, ndts, ntwd = global_parameters()
    # Load CHIRPS data
    year_data_end = (datetime(year_clm_end,month_start,day_start)+timedelta(days=ndts+1)).year
    requested_years = [*range(year_clm_start, year_data_end+1)]
    available_years = [year for year in requested_years if path.exists(f'{chirps_dir}/chirps-v2.0.{year}.days_p25.nc')]
    years_clm = list(set([*range(year_clm_start, year_clm_end+1)]).intersection(available_years))
    nyrs = len(years_clm)
    if len(available_years) < len(requested_years):
        missing_years = ' '.join(map(str, list(set(requested_years).difference(set(available_years)))))
        warnings.warn(f"The following years of CHIRPS data could not be loaded:"+"\n"+missing_years)
    file_list = [f'{chirps_dir}/chirps-v2.0.{year}.days_p25.nc' for year in available_years]
    partial_func = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
    print("Loading data ...")
    ds = xr.open_mfdataset(file_list, concat_dim='time', preprocess=partial_func, combine='nested', combine_attrs='drop_conflicts')
    nlon = ds.lon.size
    nlat = ds.lat.size
    prcp_1d =  np.full((nyrs,ndts,nlat,nlon), np.nan, dtype=float)
    prcp_3d =  np.full((nyrs,ndts,nlat,nlon), np.nan, dtype=float)
    for iyr in range(nyrs):
        date_start = datetime(years_clm[iyr],month_start,day_start)
        date_end_1d = date_start + timedelta(days=ndts-1)
        date_end_3d = date_start + timedelta(days=ndts+1)
        prcp_1d[iyr,:,:,:] = ds.sel(time=slice(date_start,date_end_1d)).precip.values
        prcp_3d[iyr,:,:,:] = ds.sel(time=slice(date_start,date_end_3d)).rolling(time=3).sum().precip.values[2:,:,:]
    ds.close()
    if not isinstance(thr_dry, np.ndarray):
        thr_dry = np.full((nlat,nlon), thr_dry, dtype=float)
    if not isinstance(thr_wet, np.ndarray):
        thr_wet = np.full((nlat,nlon), thr_wet, dtype=float)
    if not isinstance(wnd_dry_spell, np.ndarray):
        wnd_dry_spell = np.full((nlat,nlon), wnd_dry_spell, dtype=int)
    if not isinstance(len_dry_spell, np.ndarray):
        len_dry_spell = np.full((nlat,nlon), len_dry_spell, dtype=int)
    exc_1d = np.greater(prcp_1d, thr_dry[None,None,:,:])
    exc_3d = np.greater(prcp_3d, thr_wet[None,None,:,:])
    onset_day = np.full((nyrs,nlat,nlon), np.nan, dtype=float)
    for iyr in range(nyrs):
        print(f"Calculating rainy season onset dates for {years_clm[iyr]} ...")
        for ilat in range(nlat):
            for ilon in range(nlon):
                if np.isnan(thr_dry[ilat,ilon]) or np.isnan(thr_wet[ilat,ilon]):
                    continue
                if wnd_dry_spell[ilat,ilon] < 3 or wnd_dry_spell[ilat,ilon] > 21:
                    print("Warning! Invalid value for 'wnd_dry_spell'. Please choose a value between 3 and 21.")
                    continue
                if len_dry_spell[ilat,ilon] < 2 or len_dry_spell[ilat,ilon] >= wnd_dry_spell[ilat,ilon]:
                    print("Warning! Invalid value for 'len_dry_spell'. Calculation of rainy season onset date not possible.")
                    continue
                if not np.any(np.isnan(prcp_1d[iyr,:,ilat,ilon])):
                    onset_day[iyr,ilat,ilon] = find_onset_day(exc_1d[iyr,:,ilat,ilon], exc_3d[iyr,:,ilat,ilon], wnd_dry_spell[ilat,ilon], len_dry_spell[ilat,ilon])
    return onset_day



# Helper function that calculates the relative frequency of days with precipitation amounts below a given threshold

def calculate_prob_below_threshold(prcp_acc, thresh):
    day_start, nwks, ndts, ntwd = global_parameters()
    prob_below_thr = np.full((ndts,prcp_acc.shape[2],prcp_acc.shape[3]), np.nan, dtype=float)
    for idt in range(ndts):
        iltl = day_start + idt - (ntwd-1)//2
        iltu = day_start + idt + (ntwd+1)//2
        prob_below_thr[idt,:,:] = np.mean(prcp_acc[:,iltl:iltu,:,:]<=thresh[None,None,:,:], axis=(0,1))
        mask = np.logical_or(np.all(np.isnan(prcp_acc[:,iltl:iltu,:,:]), axis=(0,1)), np.isnan(thresh))
        prob_below_thr[idt,:,:][mask] = np.nan
    return prob_below_thr



# Function that bilinearly interpolates the forecasts and uses their CDF for adjusting (quantile mapping) the original thresholds

def interpolate_and_map_threshold(prcp_fcst, prob_below_thr, lat_fcst, lon_fcst, lat_trgt, lon_trgt):
    nyrs, nmbs, nlt, nlatf, nlonf = prcp_fcst.shape
    nlat = len(lat_trgt)
    nlon = len(lon_trgt)
    day_start, nwks, ndts, ntwd = global_parameters()
    prob_qt = np.arange(1,nyrs+1)/(nyrs+1)                                     # Quantile levels to reduce the sample of size nyrs x nmbs to one of size nyrs
    prcp_fcst_sample = np.full((ntwd,nyrs,nlat,nlon), np.nan, dtype=float)     # Moving window sample of forecast precipitation accumulations
    thresh_adj = np.full((ndts,nlat,nlon), np.nan, dtype=float)
    mask = np.any(np.isnan(prob_below_thr), axis=0)
    for itwd in range(ntwd-1):
        ilt = day_start + itwd - (ntwd-1)//2
        if np.all(np.isnan(prcp_fcst[:,:,ilt,:,:])):
            continue
        prcp_fcst_ip = interpolate_forecasts(prcp_fcst[:,:,ilt,:,:], lat_fcst, lon_fcst, lat_trgt, lon_trgt)
        prcp_fcst_sample[itwd,:,:,:] = np.nanquantile(prcp_fcst_ip, axis=(0,1), q=prob_qt)
    for idt in range(ndts):
        if day_start + idt >= nlt:                         # date outside the range of available lead times
            thresh_adj[idt,:,:] = thresh_adj[idt-1,:,:]    #  -> constant extrapolation of adjusted thresholds
            continue
        ilt = day_start + idt + (ntwd-1)//2
        itwd = (ntwd-1+idt) % ntwd            # index for the slice of the time window to be overwritten with new data        
        if ilt < nlt:
            prcp_fcst_ip = interpolate_forecasts(prcp_fcst[:,:,ilt,:,:], lat_fcst, lon_fcst, lat_trgt, lon_trgt)
            prcp_fcst_sample[itwd,:,:,:] = np.nanquantile(prcp_fcst_ip, axis=(0,1), q=prob_qt)
        else:
            prcp_fcst_sample[itwd,:,:,:] = np.nan     # cut off time window at the end
        for ilat in range(nlat):
            for ilon in range(nlon):
                if mask[ilat,ilon]:
                    continue
                prcp_fcst_pct = np.nanpercentile(prcp_fcst_sample[:,:,ilat,ilon], q=range(1,100,1))
                itp_fct = interp1d(np.linspace(0.,.99,100), np.append(0.,prcp_fcst_pct), kind='linear', bounds_error=False, fill_value=(0.,prcp_fcst_pct[-1]), assume_sorted=True)
                thresh_adj[idt,ilat,ilon] = itp_fct(prob_below_thr[idt,ilat,ilon])
    return thresh_adj



# Main function for threshold adjustment: loads CHIRPS and forecast data, calculates climatological exceedance probabilities, and calls the helper function above

def calculate_adjusted_thresholds(region, month_start, year_clm_start, year_clm_end, system, thr_dry, thr_wet, chirps_dir, fcst_dir, cv=False):
    lon_bnds, lat_bnds = domain_boundaries(region)
    day_start, nwks, ndts, ntwd = global_parameters()
   # Load CHIRPS data and calculate 1-day/3-day precipitation amounts and their climatological probabilities not exceeding the chosen threshold values
    year_data_end = (datetime(year_clm_end,month_init_dict[month_start],day_start)+timedelta(days=ndts+1)).year
    requested_years = [*range(year_clm_start, year_data_end+1)]
    available_years = [year for year in requested_years if path.exists(f'{chirps_dir}/chirps-v2.0.{year}.days_p25.nc')]
    years_clm = list(set([*range(year_clm_start, year_clm_end+1)]).intersection(available_years))
    nyrs = len(years_clm)
    if len(available_years) < len(requested_years):
        missing_years = ' '.join(map(str, list(set(requested_years).difference(set(available_years)))))
        warnings.warn(f"The following years of CHIRPS data could not be loaded:"+"\n"+missing_years)
    file_list = [f'{chirps_dir}/chirps-v2.0.{year}.days_p25.nc' for year in available_years]
    partial_func = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
    print("Loading CHIRPS data ...")
    ds = xr.open_mfdataset(file_list, concat_dim='time', preprocess=partial_func, combine='nested', combine_attrs='drop_conflicts')
    lon_chirps = ds.lon.values
    lat_chirps = ds.lat.values
    nlon = ds.lon.size
    nlat = ds.lat.size
    prcp_1d =  np.full((nyrs,ndts+ntwd+1,nlat,nlon), np.nan, dtype=float)
    prcp_3d =  np.full((nyrs,ndts+ntwd+1,nlat,nlon), np.nan, dtype=float)
    for iyr in range(nyrs):
        date_start = datetime(years_clm[iyr],month_init_dict[month_start],1)
        date_end_1d = date_start + timedelta(days=ndts+ntwd)
        date_end_3d = date_start + timedelta(days=ndts+ntwd+2)
        prcp_1d[iyr,:,:,:] = ds.sel(time=slice(date_start,date_end_1d)).precip.values
        prcp_3d[iyr,:,:,:] = ds.sel(time=slice(date_start,date_end_3d)).rolling(time=3).sum().precip.values[2:,:,:]
    ds.close()
    if not isinstance(thr_dry, np.ndarray):
        thr_dry = np.full((nlat,nlon), thr_dry, dtype=float)
    if not isinstance(thr_wet, np.ndarray):
        thr_wet = np.full((nlat,nlon), thr_wet, dtype=float)
   # Load hindcast data and calculate 1-day/3-day precipitation amounts
    requested_years = [*range(year_clm_start, year_clm_end+1)]
    available_years = [year for year in requested_years if path.exists(get_filename(fcst_dir, system, year, month_start))]
    if len(available_years) < len(requested_years):
        missing_years = ' '.join(map(str, list(set(requested_years).difference(set(available_years)))))
        warnings.warn(f"The following years of {system.upper()} forecast data could not be loaded:"+"\n"+missing_years)
    if len(available_years) == 0:
        raise Exception("No forecast data found to calculate percentiles.")
    file_list = [get_filename(fcst_dir, system, year, month_start) for year in available_years]
    ds = xr.open_mfdataset(file_list[0], preprocess=partial_func)
    lon_fcst = ds.lon.values
    lat_fcst = ds.lat.values
    nmbs, nlts, nlatf, nlonf = ds.precip.shape
    ds.close()
    prcp_fcst_1d = np.full((nyrs,nmbs,nlts-2,nlatf,nlonf), np.nan, dtype=np.float32)
    prcp_fcst_3d = np.full((nyrs,nmbs,nlts-2,nlatf,nlonf), np.nan, dtype=np.float32)
    print(f"Loading {system.upper()} forecast data ...")
    for iyr in range(len(available_years)):
        if system[:3] == 'wrf':
            ds = xr.open_mfdataset(file_list[iyr], preprocess=partial_func)
            prcp_fcst_1d[iyr,:,:,:] = ds.precip.values[:,:-2,:,:]
            prcp_fcst_3d[iyr,:,:,:] = ds.rolling(valid_time=3).sum().precip.values[:,2:,:,:]
            ds.close()
        else:
            ds = xr.open_mfdataset(file_list[iyr], preprocess=partial_func)
            prcp_fcst_cum = 1e3*np.insert(ds.precip.values[:nmbs,:nlts,:,:], 0, 0.0, axis=1)    # add zero accumulation at lead time 0
            ds.close()
            prcp_fcst_1d[iyr,:,:,:] = np.maximum(0., prcp_fcst_cum[:,1:-2,:,:]-prcp_fcst_cum[:,:-3,:,:])
            prcp_fcst_3d[iyr,:,:,:] = np.maximum(0., prcp_fcst_cum[:,3:,:,:]-prcp_fcst_cum[:,:-3,:,:])
   # Compose a moving window forecast sample, estimate model climatology, and use to quantile-map the threshold values
    print("Calculating adjusted dry spell thresholds ...")
    if cv:
        thr_dry_adj = np.full((nyrs,ndts,nlat,nlon), np.nan, dtype=float)
        for iyr in range(nyrs):
            print(f"Processing year {years_clm[iyr]}")
            prcp_1d_cv = np.delete(prcp_1d, iyr, axis=0)
            prcp_1d_pb_thr_dry = calculate_prob_below_threshold(prcp_1d_cv, thr_dry)
            prcp_fcst_1d_cv = np.delete(prcp_fcst_1d, iyr, axis=0)
            thr_dry_adj[iyr,:,:,:] = np.minimum(thr_dry[None,:,:]*5, np.maximum(interpolate_and_map_threshold(prcp_fcst_1d_cv, prcp_1d_pb_thr_dry, lat_fcst, lon_fcst, lat_chirps, lon_chirps), thr_dry[None,:,:]/5))
    else:
        prcp_1d_pb_thr_dry = calculate_prob_below_threshold(prcp_1d, thr_dry)
        thr_dry_adj = np.minimum(thr_dry[None,:,:]*5, np.maximum(interpolate_and_map_threshold(prcp_fcst_1d, prcp_1d_pb_thr_dry, lat_fcst, lon_fcst, lat_chirps, lon_chirps), thr_dry[None,:,:]/5))
    print("Calculating adjusted wet spell thresholds ...")
    if cv:
        thr_wet_adj = np.full((nyrs,ndts,nlat,nlon), np.nan, dtype=float)
        for iyr in range(nyrs):
            print(f"Processing year {years_clm[iyr]}")
            prcp_3d_cv = np.delete(prcp_3d, iyr, axis=0)
            prcp_3d_pb_thr_wet = calculate_prob_below_threshold(prcp_3d_cv, thr_wet)
            prcp_fcst_3d_cv = np.delete(prcp_fcst_3d, iyr, axis=0)
            thr_wet_adj[iyr,:,:,:] = np.minimum(thr_wet[None,:,:]*5, np.maximum(interpolate_and_map_threshold(prcp_fcst_3d_cv, prcp_3d_pb_thr_wet, lat_fcst, lon_fcst, lat_chirps, lon_chirps), thr_wet[None,:,:]/5))
    else:
        prcp_3d_pb_thr_wet = calculate_prob_below_threshold(prcp_3d, thr_wet)
        thr_wet_adj = np.minimum(thr_wet[None,:,:]*5, np.maximum(interpolate_and_map_threshold(prcp_fcst_3d, prcp_3d_pb_thr_wet, lat_fcst, lon_fcst, lat_chirps, lon_chirps), thr_wet[None,:,:]/5))
    return thr_dry_adj, thr_wet_adj



# Function for calculating a rainy season onset date based on a new set of ensemble forecasts and adjusted thresholds

def calculate_onset_fcst(region, month_start, year_fcst, system, thresh_dry, thresh_wet, wnd_dry_spell, len_dry_spell, lat_trgt, lon_trgt, fcst_dir):
    lon_bnds, lat_bnds = domain_boundaries(region)
    day_start, nwks, ndts, ntwd = global_parameters()
    nlat = len(lat_trgt)
    nlon = len(lon_trgt)
   # Load file with new forecasts
    filename = get_filename(fcst_dir, system, year_fcst, month_start)
    if not path.exists(filename):
        raise Exception(f"No forecast data found for selected year {year_fcst}.")
    print(f"Loading and interpolating forecast data for year {year_fcst} ...")
    partial_func = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
    if system[:3] == 'wrf':
        ds = xr.open_mfdataset(filename, preprocess=partial_func)
        lon_fcst = ds.lon.values
        lat_fcst = ds.lat.values
        nmbs = ds.init_date.size
        nlts = ds.valid_time.size
        if nlts < day_start + ndts - 1:
            prcp_fcst_1d = np.pad(ds.precip.values[:,(day_start-1):,:,:], ((0,0),(0,day_start+ndts-1-nlts),(0,0),(0,0)), mode='constant', constant_values=np.nan)      # pad with NaNs at the end
        else:
            prcp_fcst_1d = ds.precip.values[:,(day_start-1):(day_start+ndts-1),:,:]
        if nlts < day_start + ndts + 1:
            prcp_fcst_3d = np.pad(ds.rolling(valid_time=3).sum().precip.values[:,(day_start+1):,:,:], ((0,0),(0,day_start+ndts+1-nlts),(0,0),(0,0)), mode='constant', constant_values=np.nan)
        else:
            prcp_fcst_3d = ds.rolling(valid_time=3).sum().precip.values[:,(day_start+1):(day_start+ndts+1),:,:]
        ds.close()
    else:
        ds = xr.open_mfdataset(filename, preprocess=partial_func)
        lon_fcst = ds.lon.values
        lat_fcst = ds.lat.values
        prcp_fcst_cum = 1e3*np.insert(ds.precip.values, 0, 0.0, axis=1)
        ds.close()
        prcp_fcst_1d = np.maximum(0., prcp_fcst_cum[:,day_start:(day_start+ndts),:,:]-prcp_fcst_cum[:,(day_start-1):(day_start+ndts-1),:,:])         # assumes that forecasts are available for these lead times
        prcp_fcst_3d = np.maximum(0., prcp_fcst_cum[:,(day_start+2):(day_start+ndts+2),:,:]-prcp_fcst_cum[:,(day_start-1):(day_start+ndts-1),:,:])
        nmbs = prcp_fcst_cum.shape[0]
   # Interpolate forecasts and record exceedances of the 1-day/3-day threshold
    prcp_fcst_1d_ip = interpolate_forecasts(prcp_fcst_1d, lat_fcst, lon_fcst, lat_trgt, lon_trgt)
    exc_1d = np.where(np.isnan(prcp_fcst_1d_ip), True, np.greater(prcp_fcst_1d_ip, thresh_dry[None,:,:,:]))   # assume a rainy day if data is missing ...
    prcp_fcst_3d_ip = interpolate_forecasts(prcp_fcst_3d, lat_fcst, lon_fcst, lat_trgt, lon_trgt)
    exc_3d = np.where(np.isnan(prcp_fcst_3d_ip), False, np.greater(prcp_fcst_3d_ip, thresh_wet[None,:,:,:]))  # ... but don't assume enough rainfall to qualify as a wet spell
    mask = np.logical_or(np.isnan(prcp_fcst_1d_ip), np.isnan(thresh_dry[None,:,:,:]))
   # Calculate rainy season onset forecast based on these exceedances
    print("Calculating rainy season onset dates ...")
    if not isinstance(len_dry_spell, np.ndarray):
        len_dry_spell = np.full((nlat,nlon), len_dry_spell, dtype=int)
    onset_day_fcst = np.full((nmbs,nlat,nlon), np.nan, dtype=np.float32)
    for imb in range(nmbs):
        if np.all(mask[imb,:,:,:]):
            continue
        for ilat in range(nlat):
            for ilon in range(nlon):
                if not np.all(mask[imb,:,ilat,ilon]):
                    onset_day_fcst[imb,ilat,ilon] = find_onset_day(exc_1d[imb,:,ilat,ilon], exc_3d[imb,:,ilat,ilon], wnd_dry_spell[ilat,ilon], len_dry_spell[ilat,ilon])
    return onset_day_fcst



# Function for augmenting the ensemble forecasts by considering forecasts at neighboring grid points

def augment_onset_fcst(onset_day_fcst, axis=0):
    if axis == 0:
        nmbs, nlat, nlon = onset_day_fcst.shape
        onset_day_fcst_padded = np.pad(onset_day_fcst, ((0,0),(1,1),(1,1)), mode='edge')
        onset_day_fcst_augm = np.full((9*nmbs,nlat,nlon), np.nan, dtype=float)
        for i in range(9):
            ixs, iys = (i%3), (i//3)
            onset_day_fcst_augm[i*nmbs:(i+1)*nmbs,:,:] = onset_day_fcst_padded[:,iys:(nlat+iys),ixs:(nlon+ixs)]
    elif axis == 1:
        nyrs, nmbs, nlat, nlon = onset_day_fcst.shape
        onset_day_fcst_padded = np.pad(onset_day_fcst, ((0,0),(0,0),(1,1),(1,1)), mode='edge')
        onset_day_fcst_augm = np.full((nyrs,9*nmbs,nlat,nlon), np.nan, dtype=float)
        for i in range(9):
            ixs, iys = (i%3), (i//3)
            onset_day_fcst_augm[:,i*nmbs:(i+1)*nmbs,:,:] = onset_day_fcst_padded[:,:,iys:(nlat+iys),ixs:(nlon+ixs)]
    else:
        raise Exception("Method only implemented for 'axis=0' and 'axis=1'.")
    return onset_day_fcst_augm



