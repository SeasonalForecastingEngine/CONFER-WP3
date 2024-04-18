
import numpy as np
import xarray as xr

from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d

from glp import domain_boundaries, climatological_reference_period, global_parameters



def interpolate_forecasts(prcp_fcst, lat_fcst, lon_fcst, lat_trgt, lon_trgt):
    nmbs, nlatf, nlonf = prcp_fcst.shape
    prcp_fcst_itp = np.full((nmbs,len(lat_trgt),len(lon_trgt)), np.nan, dtype=np.float32)
    for imb in range(nmbs):
        if np.all(np.isnan(prcp_fcst[imb,:,:])):
            continue
        itpfct = RectBivariateSpline(lat_fcst, lon_fcst, prcp_fcst[imb,:,:], kx=1, ky=1, s=0)
        prcp_fcst_itp[imb,:,:] = itpfct.__call__(lat_trgt, lon_trgt, grid=True)
    return prcp_fcst_itp


def find_onset_day(exc1d, exc3d, len_dry_spell):
    n = len(exc1d)
    wet_spell = np.logical_and(exc3d[:(n-2)], exc1d[:(n-2)])
    ind_dry_spell = np.expand_dims(np.arange(len_dry_spell),0)+np.expand_dims(np.arange(n-len_dry_spell+1),0).T
    dry_spell = np.all(~exc1d[ind_dry_spell], axis=1)
    ind_dry_spell_wnd = np.expand_dims(np.arange(22-len_dry_spell),0)+np.expand_dims(np.arange(n-23),0).T
    onset = np.logical_and(wet_spell[:(n-23)], ~np.any(dry_spell[3:][ind_dry_spell_wnd], axis=1))
    onset_day = -1 if np.all(~onset) else np.nonzero(onset)[0][0]+1
    return onset_day


def calculate_onset_hist(region, month_start, thr_dry, thr_wet, len_dry_spell, data_dir):
    lon_bounds, lat_bounds = domain_boundaries(region)
    year_clm_start, year_clm_end = climatological_reference_period()
    day_start, nwks, ndts, ntwd = global_parameters()
    # Load CHIRPS data
    filename_chirps = f'{data_dir}CHIRPS_daily_{year_clm_start}-{year_clm_end}_{month_start}.nc'
    data_load = xr.open_dataset(filename_chirps, engine='netcdf4')
    data_subset = data_load.sel(lat=slice(lat_bounds[0],lat_bounds[1]), lon=slice(lon_bounds[0],lon_bounds[1]))
    prcp_daily = data_subset.prcp.values
    data_load.close()
    nyrs, nlt, nlat, nlon = prcp_daily.shape    
    prcp_1d = prcp_daily[:,(day_start-1):(day_start+ndts-1),:,:]
    prcp_3d = prcp_daily[:,(day_start-1):(day_start+ndts-1),:,:] + prcp_daily[:,day_start:(day_start+ndts),:,:] + prcp_daily[:,(day_start+1):(day_start+ndts+1),:,:]
    exc_1d = np.greater(prcp_1d, thr_dry[None,None,:,:])
    exc_3d = np.greater(prcp_3d, thr_wet[None,None,:,:])
    onset_day = np.full((nyrs,nlat,nlon), np.nan, dtype=float)
    for iyr in range(nyrs):
        for ilat in range(nlat):
            for ilon in range(nlon):
                if len_dry_spell[ilat,ilon] < 2 or len_dry_spell[ilat,ilon] > 20:
                    print("Warning! Invalid value for 'len_dry_spell'. Calculation of rainy season onset date not possible.")
                    continue
                if not np.any(np.isnan(prcp_1d[iyr,:,ilat,ilon])):
                    onset_day[iyr,ilat,ilon] = find_onset_day(exc_1d[iyr,:,ilat,ilon], exc_3d[iyr,:,ilat,ilon], len_dry_spell[ilat,ilon])
    return onset_day


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
        prcp_fcst_ip = np.full((nyrs,nmbs,nlat,nlon), np.nan, dtype=float)
        for iyr in range(nyrs):
            prcp_fcst_ip[iyr,:,:,:] = interpolate_forecasts(prcp_fcst[iyr,:,ilt,:,:], lat_fcst, lon_fcst, lat_trgt, lon_trgt)
        prcp_fcst_sample[itwd,:,:,:] = np.nanquantile(prcp_fcst_ip, axis=(0,1), q=prob_qt)
    for idt in range(ndts):
        ilt = day_start + idt + (ntwd-1)//2
        itwd = (ntwd-1+idt) % ntwd           # index for the slice of the time window to be overwritten with new data
        for iyr in range(nyrs):
            prcp_fcst_ip[iyr,:,:,:] = interpolate_forecasts(prcp_fcst[iyr,:,ilt,:,:], lat_fcst, lon_fcst, lat_trgt, lon_trgt)
        prcp_fcst_sample[itwd,:,:,:] = np.nanquantile(prcp_fcst_ip, axis=(0,1), q=prob_qt)
        for ilat in range(nlat):
            for ilon in range(nlon):
                if mask[ilat,ilon]:
                    continue
                prcp_fcst_pct = np.nanpercentile(prcp_fcst_sample[:,:,ilat,ilon], q=range(1,100,1))
                itp_fct = interp1d(np.linspace(0.,.99,100), np.append(0.,prcp_fcst_pct), kind='linear', bounds_error=False, fill_value=(0.,prcp_fcst_pct[-1]), assume_sorted=True)
                thresh_adj[idt,ilat,ilon] = itp_fct(prob_below_thr[idt,ilat,ilon])
    return thresh_adj


def calculate_adjusted_thresholds(region, month_start, system, thr_dry, thr_wet, data_dir):
    lon_bounds, lat_bounds = domain_boundaries(region)
    year_clm_start, year_clm_end = climatological_reference_period()
   # Load CHIRPS data and calculate 1-day/3-day precipitation amounts and their climatological probabilities not exceeding the chosen threshold values
    filename_chirps = f'{data_dir}CHIRPS_daily_{year_clm_start}-{year_clm_end}_{month_start}.nc'
    data_load = xr.open_dataset(filename_chirps, engine='netcdf4')
    data_subset = data_load.sel(lat=slice(lat_bounds[0],lat_bounds[1]), lon=slice(lon_bounds[0],lon_bounds[1]))
    lon_chirps = data_subset.lon.values
    lat_chirps = data_subset.lat.values
    prcp_daily = data_subset.prcp.values
    data_load.close()
    prcp_1d = prcp_daily[:,:-2,:,:]
    prcp_3d = prcp_daily[:,:-2,:,:] + prcp_daily[:,1:-1,:,:] + prcp_daily[:,2:,:,:]
    prcp_1d_pb_thr_dry = calculate_prob_below_threshold(prcp_1d, thr_dry)
    prcp_3d_pb_thr_wet = calculate_prob_below_threshold(prcp_3d, thr_wet)
   # Load hindcast data and calculate 1-day/3-day precipitation amounts
    filename_ensfcst = f'{data_dir}{system.upper()}_daily_{year_clm_start}-{year_clm_end}_{month_start}.nc'
    data_load = xr.open_dataset(filename_ensfcst, engine='netcdf4')
    data_subset = data_load.sel(lat=slice(lat_bounds[0],lat_bounds[1]), lon=slice(lon_bounds[0],lon_bounds[1]))
    lon_fcst = data_subset.lon.values
    lat_fcst = data_subset.lat.values
    prcp_fcst_cum = np.insert(data_subset.prcp.values, 0, 0.0, axis=2)    # add zero accumulation at lead time 0
    data_load.close()
    prcp_fcst_1d = prcp_fcst_cum[:,:,1:-2,:,:] - prcp_fcst_cum[:,:,:-3,:,:]
    prcp_fcst_3d = prcp_fcst_cum[:,:,3:,:,:] - prcp_fcst_cum[:,:,:-3,:,:]
   # Compose a moving window forecast sample, estimate model climatology, and use to quantile-map the threshold values
    thr_dry_adj = interpolate_and_map_threshold(prcp_fcst_1d, prcp_1d_pb_thr_dry, lat_fcst, lon_fcst, lat_chirps, lon_chirps)
    thr_wet_adj = interpolate_and_map_threshold(prcp_fcst_3d, prcp_3d_pb_thr_wet, lat_fcst, lon_fcst, lat_chirps, lon_chirps)
    return thr_dry_adj, thr_wet_adj


def calculate_onset_fcst(region, month_start, year_fcst, system, thresh_dry, thresh_wet, len_dry_spell, lat_trgt, lon_trgt, data_dir):
    lon_bounds, lat_bounds = domain_boundaries(region)
    day_start, nwks, ndts, ntwd = global_parameters()
    nlat = len(lat_trgt)
    nlon = len(lon_trgt)
   # Load file with new forecasts
    filename_fcst = f'{data_dir}{system.upper()}_daily_{year_fcst}_{month_start}.nc'
    data_load = xr.open_dataset(filename_fcst, engine='netcdf4')
    data_subset = data_load.sel(lat=slice(lat_bounds[1],lat_bounds[0]), lon=slice(lon_bounds[0],lon_bounds[1]))
    lon_fcst = data_subset.lon.values
    lat_fcst = data_subset.lat.values[::-1]
    prcp_fcst_cum = np.insert(data_subset.prcp.values[:,:,::-1,:], 0, 0.0, axis=1)    # add zero accumulation at lead time 0
    data_load.close()
    prcp_fcst_1d = prcp_fcst_cum[:,day_start:(day_start+ndts),:,:] - prcp_fcst_cum[:,(day_start-1):(day_start+ndts-1),:,:]
    prcp_fcst_3d = prcp_fcst_cum[:,(day_start+2):(day_start+ndts+2),:,:] - prcp_fcst_cum[:,(day_start-1):(day_start+ndts-1),:,:]
   # Interpolate forecasts and record exceedances of the 1-day/3-day threshold
    nmbs = prcp_fcst_cum.shape[0]
    exc_1d = np.zeros((ndts,nmbs,nlat,nlon), dtype=bool)
    exc_3d = np.zeros((ndts,nmbs,nlat,nlon), dtype=bool)
    mask = np.zeros((ndts,nmbs,nlat,nlon), dtype=bool)
    for idt in range(ndts):
        prcp_fcst_1d_ip = interpolate_forecasts(prcp_fcst_1d[:,idt,:,:], lat_fcst, lon_fcst, lat_trgt, lon_trgt)
        exc_1d[idt,:,:,:] = np.greater(prcp_fcst_1d_ip, thresh_dry[idt,None,:,:])
        prcp_fcst_3d_ip = interpolate_forecasts(prcp_fcst_3d[:,idt,:,:], lat_fcst, lon_fcst, lat_trgt, lon_trgt)
        exc_3d[idt,:,:,:] = np.greater(prcp_fcst_3d_ip, thresh_wet[idt,None,:,:])
        mask[idt,:,:,:] = np.logical_or(np.isnan(prcp_fcst_1d_ip), np.isnan(thresh_dry[idt,None,:,:]))
   # Calculate rainy season onset forecast based on these exceedances
    onset_day_fcst = np.full((nmbs,nlat,nlon), np.nan, dtype=np.float32)
    for imb in range(nmbs):
        if np.all(mask[:,imb,:,:]):
            continue
        for ilat in range(nlat):
            for ilon in range(nlon):
                if not np.any(mask[:,imb,ilat,ilon]):
                    onset_day_fcst[imb,ilat,ilon] = find_onset_day(exc_1d[:,imb,ilat,ilon], exc_3d[:,imb,ilat,ilon], len_dry_spell[ilat,ilon])
    return onset_day_fcst


