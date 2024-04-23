import os
import sys

import numpy as np
import xarray as xr

from os import path

chirps_dir = sys.argv[1]

#chirps_dir = '/media/datadisk/pro/SFE/CHIRPS_pentad_nc/'

filenames_all = sorted(os.listdir(chirps_dir))

for filename in filenames_all:
    if filename[:6] != 'chirps' or filename[-10:] != 'pentads.nc':
        continue
    if path.exists(f'{chirps_dir}{filename[:-3]}_p25.nc'):
        print(f"file '{filename}' already exists.")
        continue
    da_prcp = xr.open_dataset(chirps_dir+filename)
    prcp = da_prcp.precip.values
    time = da_prcp.time.values
    lat = da_prcp.latitude.values
    lon = da_prcp.longitude.values
    da_prcp.close()
    npts, nlat, nlon = prcp.shape
    nlat_upsc = nlat // 5
    nlon_upsc = nlon // 5
    prcp_upsc = np.full((npts,nlat_upsc,nlon_upsc), np.nan, dtype=float)
    for iy, jy in zip(range(nlat_upsc), range(2,nlat,5)):
        for ix, jx in zip(range(nlon_upsc), range(2,nlon,5)):
            if np.sum(np.all(np.isnan(prcp[:,(jy-2):(jy+3),(jx-2):(jx+3)]), axis=0)) < 10:
                prcp_upsc[:,iy,ix] = np.nanmean(prcp[:,(jy-2):(jy+3),(jx-2):(jx+3)], axis=(1,2))
    print(f"saving upscaled output as '{filename[:-3]}_p25.nc'.")
    da_prcp_pct = xr.DataArray(
        data= prcp_upsc,
        dims=['time','latitude','longitude'],
        coords={'time': time, 'latitude': lat[2::5], 'longitude': lon[2::5],},
        name='precip',
        attrs=dict(
            description='convective precipitation rate',
            units='mm/pentad',),
        )
    da_prcp_pct.to_netcdf(f'{chirps_dir}{filename[:-3]}_p25.nc')



