import os
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from os import path

fcst_dir = sys.argv[1]
dom = sys.argv[2]
month_init = sys.argv[3]

#fcst_dir = '/home/michael/nr/samba/PostClimDataNoBackup/CONFER/Data/Forecasts_daily_nc/'
#dom = 'd02'
#month_init = 'may'

dates_init = {'may': ['050100', '051100', '051600', '052100']}[month_init]
date_start = {'may': '05-01'}[month_init]
date_end = {'may': '10-15'}[month_init]
season = {'feb': 'MAM', 'may': 'JJAS', 'aug': 'OND'}[month_init]

years = [*range(1991,2024)]

lon_bnds, lat_bnds = [22, 52], [-12, 18]

for year in years:
    date_range = pd.date_range(f'{year}-{date_start}', f'{year}-{date_end}')
    ndts = date_range.size
    prcp = np.full((4,ndts,120,120), np.nan, dtype=float)
    for iidt, date_init in zip(range(4), dates_init):
        filename = f'{fcst_dir}wrf_{dom}/{month_init}/{season}_WRF_{date_init}/WRF_{season}_DailyPrec_init_025g_{year}{date_init}_trop_{dom}.nc'
        if not path.exists(filename):
            warnings.warn(f"{filename} not found.")
            continue
        data_load = xr.open_dataset(filename, engine='netcdf4').sel(lon=slice(*lon_bnds), lat=slice(*lat_bnds))
        time = data_load.Time.values
        for idt in range(ndts):
            idt_matched = np.where(date_range[idt]==time)[0]
            if idt_matched.size > 0:
                prcp[iidt,idt,:,:] = data_load.prec.values[idt_matched[0],:,:]
        data_load.close()
    
    da_prcp = xr.DataArray(
        data= prcp,
        dims=['init_date','valid_time','latitude','longitude'],
        coords={'init_date': dates_init, 'valid_time': date_range, 'latitude': data_load.lat.values, 'longitude': data_load.lon.values,},
        name='precip',
        attrs=dict(
            description='forecast daily precipitation amount',
            units='mm/m^2',),
        )
    da_prcp.to_netcdf(f'{fcst_dir}wrf_{dom}/{month_init}/wrf_{dom}_{month_init}_{year}.nc')



