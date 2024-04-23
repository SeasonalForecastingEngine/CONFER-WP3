
import numpy as np
import pandas as pd
import xarray as xr

from scipy.stats  import norm


def calculate_local_stdv(season, period_clm, anomaly_dir):      # Should be calculated when calculating the anomalies and saved out in EOF files 
    period_clm_str = f'{period_clm[0]}-{period_clm[1]}'
    nc = xr.open_dataset(f'{anomaly_dir}refper_{period_clm_str}/precip_full_{season}.nc', engine='netcdf4')
    nc_subset = nc.sel(year=slice(period_clm[0],period_clm[1]), loy=period_clm[0])
    #year = nc_subset.year.values
    #lat = nc_subset.lat.values
    #lon = nc_subset.lon.values
    #prcp_tercile_cat = nc_subset.tercile_cat.values   # dimension: (lat, lon, year, loy)
    prcp_ano = nc_subset.ano_norm.values
    nc.close()
    return np.std(prcp_ano, axis=2, ddof=1)



def calculate_tercile_probability_forecasts(season, year_fcst, month_init, period_train, period_clm, indices_dir, anomaly_dir, eof_dir, fcst_dir):
    ntg = 7 # global parameter ??
    period_clm_str = f'{period_clm[0]}-{period_clm[1]}'
    period_train_str = f'{period_train[0]}-{period_train[1]}'
    scaling = calculate_local_stdv(season, period_clm, anomaly_dir)
    # Load (first 7) EOFs and calculate residual variance
    nc = xr.open_dataset(f'{eof_dir}refper_{period_clm_str}/prec_full_seasonal_{season}.nc')
    nc_subset = nc.sel(loy=period_clm[0], eof=slice(1,ntg))
    eofs = np.transpose(nc_subset.u.values, axes=(0,2,1))
    eigenvalues = (nc_subset.d.values**2) / (period_clm[1]-period_clm[0])
    nc.close()
    var_eps = scaling**2 - np.sum(eigenvalues[:,None,None]*eofs**2, axis=0)    # climatological variance minus variance explained by the first 7 EOFs
    # Load estimated coefficients and covariance matrix of the prediction errors of the factor loadings
    coefficients = pd.read_csv(f'{fcst_dir}refper_{period_clm_str}_cvper_{period_train_str}/coefficients_indices_lasso_full_im{month_init}_{season}.csv', index_col=0)
    fl_eof_cov = pd.read_csv(f'{fcst_dir}refper_{period_clm_str}_cvper_{period_train_str}/fls_cov_indices_lasso_full_im{month_init}_{season}.csv', index_col=0).loc[period_clm[1]].to_numpy().reshape(ntg,ntg)
    # Load indices for forecast year
    ts_indices = pd.Series(index=coefficients.columns)
    for index_name in coefficients.columns[1:]:
        filename_index = f'{indices_dir}refper_{period_clm_str}/indices/{index_name}_full.csv'
        ts_indices[index_name] = pd.read_csv(filename_index, index_col=['year','month','loy'], usecols=['year','month','loy','fl']).rename(columns={'fl':index_name}).loc[(year_fcst,month_init-1,period_clm[1]),:].iloc[0]
    ts_indices['year'] = (year_fcst-2000)/10
    # Calculate predictive mean of factor loadings
    fl_eof_mean = coefficients.dot(ts_indices).to_numpy()
    ## Calculate mean and variance of the probabilistic forecast in normal space
    mean_ml = np.sum(fl_eof_mean[:,None,None]*eofs, axis=0)
    var_ml = np.sum(np.sum(fl_eof_cov[:,:,None,None]*eofs[None,:,:,:], axis=1)*eofs, axis=0) + var_eps
    mean_ml_stdz = mean_ml / scaling
    stdv_ml_stdz = np.sqrt(var_ml) / scaling
    ## Calculate tercile forecasts
    prob_bn = norm.cdf((norm.ppf(0.333)-mean_ml_stdz)/stdv_ml_stdz)
    prob_an = 1.-norm.cdf((norm.ppf(0.667)-mean_ml_stdz)/stdv_ml_stdz)
    return prob_bn, prob_an









