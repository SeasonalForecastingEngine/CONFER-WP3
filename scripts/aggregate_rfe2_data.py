
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

from os import path
from datetime import date, datetime, timedelta
from osgeo import gdal

rfe2_daily_dir = sys.argv[1]
rfe2_pentad_dir = sys.argv[2]

#rfe2_daily_dir = '/home/michael/nr/samba/PostClimDataNoBackup/CONFER/Data/RFE2_daily/'
#rfe2_pentad_dir = '/home/michael/nr/samba/PostClimDataNoBackup/CONFER/Data/RFE2_pentad/'

filenames = np.sort(os.listdir(rfe2_daily_dir))
filenames_date = np.array([int(filename[11:19]) if filename[20:]=='tif' else np.nan  for filename in filenames])
nfiles = len(filenames)

years = np.unique((filenames_date//1e4).astype(int)).tolist()

dataset = gdal.Open(f'{rfe2_daily_dir}{filenames[0]}')
xoff, a, b, yoff, d, e = dataset.GetGeoTransform()
band1 = dataset.GetRasterBand(1)
prcp_tif = band1.ReadAsArray()

nlat, nlon = prcp_tif.shape
lon = xoff + a * np.arange(0.5,nlon)
lat = yoff + e * np.arange(0.5,nlat)


for year in years:
    filename_out = f'rfe2.{year}.pentads.nc'
    if path.exists(rfe2_pentad_dir+filename_out):
        print(f"file '{filename_out}' already exists.")
        continue
    prcp_pentad = np.full((72,nlat,nlon), np.nan, dtype=np.float32)
    time = []
    for im in range(12):
        dates_pentad_start = [datetime.strptime(str(year)+format(im+1,'02d')+day,"%Y%m%d") for day in ['01','06','11','16','21','26']]
        dates_pentad_end = dates_pentad_start[1:]+[datetime.strptime(str(year+(im+1)//12)+format((im+1)%12+1,'02d')+'01',"%Y%m%d")]
        time.extend(dates_pentad_start)
        for ipt in range(6):
            ndays_pentad = (dates_pentad_end[ipt]-dates_pentad_start[ipt]).days
            filename_index = np.logical_and(filenames_date>=int(dates_pentad_start[ipt].strftime("%Y%m%d")),  filenames_date<int(dates_pentad_end[ipt].strftime("%Y%m%d")))
            filenames_pentad = [filenames[i] for i in range(nfiles) if filename_index[i]]
            if len(filenames_pentad) < ndays_pentad:
                continue
            prcp_daily = np.full((ndays_pentad,nlat,nlon), np.nan, dtype=np.float32)
            for ifn in range(ndays_pentad):
                dataset = gdal.Open(f'{rfe2_daily_dir}{filenames_pentad[ifn]}')
                band1 = dataset.GetRasterBand(1)
                prcp_daily[ifn,:,:] = band1.ReadAsArray()
            prcp_pentad[im*6+ipt,:,:] = np.sum(prcp_daily, axis=0)
    print(f"saving pentad data as '{rfe2_pentad_dir}{filename_out}'.")
    da_prcp_pct = xr.DataArray(
        data= prcp_pentad,
        dims=['time','latitude','longitude'],
        coords={'time': time, 'latitude': lat, 'longitude': lon,},
        name='precip',
        attrs=dict(
            description='convective precipitation rate',
            units='mm/pentad',),
        )
    da_prcp_pct.to_netcdf(rfe2_pentad_dir+filename_out)


