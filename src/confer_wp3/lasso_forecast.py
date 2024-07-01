"""
This file contains code for technical calculations used in the lasso forecast.
"""
import numpy as np
import pandas as pd
import xarray as xr

from eofs.standard import Eof
from scipy.stats  import norm
from scipy.interpolate import interp1d


def calculate_anomalies(prec_data, year, period_clm):
    """
    Calculate anomalies in precipitation data relative to a specified climatological period.

    Parameters:
    - prec_data (numpy.ndarray): A 3D numpy array representing the precipitation data with dimensions (time, lat, lon).
    - year (numpy.ndarray): A 1D numpy array of years corresponding to the first dimension of `prec_data`.
    - period_clm (tuple): A tuple of two integers representing the start and end years of the climatological reference period.

    Returns:
    - anomalies (numpy.ndarray): A 3D numpy array of the same shape as `prec_data` representing the precipitation anomalies.

    Note:
    - The function uses `np.nanmean` to calculate the mean, ignoring any NaN values in the data. This is due to the masking of the region being done with NaNs.
    """
    # Select the reference period using a boolean mask
    ref_period_mask = (year >= period_clm[0]) & (year <= period_clm[1])
    ref_period_indices = np.where(ref_period_mask)[0]

    # Calculate climatology (mean over the reference period)
    climatology = np.nanmean(prec_data[ref_period_indices, :, :], axis=0)

    # Calculate anomalies by subtracting climatology
    anomalies = prec_data - climatology[np.newaxis, :, :]
    return anomalies
    

def quantile_mapping(data, year, period_clm):
    """
    Apply quantile mapping to transform the precipitation data based on a specified climatological period.

    Parameters:
    - data (numpy.ndarray): A 3D numpy array representing the precipitation data with dimensions (time, lat, lon).
    - year (numpy.ndarray): A 1D numpy array of years corresponding to the first dimension of `data`.
    - period_clm (tuple): A tuple of two integers representing the start and end years of the climatological reference period.

    Returns:
    - transformed_data (numpy.ndarray): A 3D numpy array of the same shape as `data` with transformed precipitation values.

    Note:
    - The function uses `np.nanstd` to calculate the standard deviation, ignoring any NaN values in the data.
    """
    # Select the reference period using a boolean mask
    ref_period_mask = (year >= period_clm[0]) & (year <= period_clm[1])
    ref_period_indices = np.where(ref_period_mask)[0]
    
    # Standard deviation for the reference period at each grid point
    ref_period_std = np.nanstd(data[ref_period_indices, :, :], axis=0)
    
    # Pre-calculate percentiles and normal scores
    n = ref_period_indices.size
    ref_percentiles = np.arange(1, n + 1) / (n + 1)
    ref_normal = norm.ppf(ref_percentiles)
    ref_min_outside_range = norm.ppf(1/(n+2))
    ref_max_outside_range = norm.ppf((n+1)/(n+2))

    # Function to add jitter
    add_jitter = lambda x: x + np.random.normal(0, 1e-10, x.shape)

    transformed_data = np.full_like(data, np.nan)

    for lat_idx in range(data.shape[1]):
        for lon_idx in range(data.shape[2]):
            # Get data in reference period and remove NaNs
            ref_data = data[ref_period_indices, lat_idx, lon_idx]
            ref_data_valid = ref_data[~np.isnan(ref_data)]
            
            if ref_data_valid.size == 0:
                continue
            
            # Add jitter to avoid identical ranks, and sort reference data
            # Add jitter again to avoid repeated values (might not be needed)
            sorted_ref_data = np.sort(add_jitter(ref_data_valid))
            sorted_ref_data_jittered = add_jitter(sorted_ref_data)

            # Interpolator for reference period quantiles
            interp_func = interp1d(sorted_ref_data_jittered, ref_normal, kind='linear', bounds_error=False, fill_value=(ref_min_outside_range, ref_max_outside_range))
            
            all_years_data = data[:, lat_idx, lon_idx]
            valid_mask = ~np.isnan(all_years_data) 
            all_years_data_valid = all_years_data[valid_mask]
            
            # Add jitter to all_years_data_valid            
            # Interpolation of normal quantiles using reference period
            all_years_normal = interp_func(add_jitter(all_years_data_valid))

            # Standardize the normal values using the reference period standard deviation
            all_years_normal_standardized = all_years_normal * ref_period_std[lat_idx, lon_idx]
            
            # Place standardized values back in the original array shape
            transformed_data[valid_mask, lat_idx, lon_idx] = all_years_normal_standardized
            
    return transformed_data


def compute_eofs_pcs(anomalies_normal, n_eofs):
    """
    Compute the Empirical Orthogonal Functions (EOFs) and Principal Components (PCs) from the normalized anomalies.

    Parameters:
    - anomalies_normal (numpy.ndarray): A 3D numpy array of normalized precipitation anomalies with dimensions (time, lat, lon).
    - n_eofs (int): The number of EOFs and PCs to compute.

    Returns:
    - eofs (numpy.ndarray): A 2D numpy array of the computed EOFs with dimensions (n_eofs, space).
    - pcs (numpy.ndarray): A 2D numpy array of the computed PCs with dimensions (time, n_eofs).
    - variance_fraction (numpy.ndarray): A 1D numpy array representing the fraction of variance explained by each EOF.
    """
    # Reshape the anomalies_normal to 2D (time, space) for EOF computation
    year, lat, lon = anomalies_normal.shape
    anomalies_normal_reshaped = anomalies_normal.reshape(year, lat * lon)

    # Mask invalid values
    anomalies_normal_masked = np.ma.masked_invalid(anomalies_normal_reshaped)
    
    # Initialize the EOF solver
    solver = Eof(anomalies_normal_masked)
    
    # Calculate the EOFs and PCs
    eofs = solver.eofs(neofs=n_eofs)
    pcs = solver.pcs(npcs=n_eofs)
    variance_fraction = solver.varianceFraction(neigs=n_eofs)
    
    return eofs, pcs, variance_fraction


def get_region_indices(region):
    """
    Retrieve the bounding box coordinates for a specified region.

    Parameters:
    - region (str): The name of the region for which to retrieve the bounding box coordinates.
      Valid region names include:
        - 'n34': Nino 3.4 region
        - 'n3': Nino 3 region
        - 'n4_1': Nino 4 (part 1) region
        - 'n4_2': Nino 4 (part 2) region
        - 'wpg': Western Pacific region
        - 'dmi_1': Dipole Mode Index (West)
        - 'dmi_2': Dipole Mode Index (East)
        - 'sji850': South Indian Ocean Jet at 850 hPa
        - 'sji200': South Indian Ocean Jet at 200 hPa
        - 'ueq850': Upper Equatorial region at 850 hPa
        - 'ueq200': Upper Equatorial region at 200 hPa
        - 'wp': Western Pacific region
        - 'wnp_1': Western North Pacific (part 1)
        - 'wnp_2': Western North Pacific (part 2)
        - 'wsp_1': Western South Pacific (part 1)
        - 'wsp_2': Western South Pacific (part 2)

    Returns:
    - dict: A dictionary containing the bounding box coordinates for the specified region, with keys:
        - 'lat_min': Minimum latitude
        - 'lat_max': Maximum latitude
        - 'lon_min': Minimum longitude
        - 'lon_max': Maximum longitude
    """
    # Define the bounding boxes for the indices
    indices_definitions = {
        'n34': {'lat_min': -5, 'lat_max': 5, 'lon_min': -170, 'lon_max': -120},
        'n3': {'lat_min': -5, 'lat_max': 5, 'lon_min': -150, 'lon_max': -90},
        'n4_1': {'lat_min': -5, 'lat_max': 5, 'lon_min': -180, 'lon_max': -150},
        'n4_2': {'lat_min': -5, 'lat_max': 5, 'lon_min': 160, 'lon_max': 180},
        'wpg': {'lat_min': 0, 'lat_max': 20, 'lon_min': 130, 'lon_max': 150},
        'dmi_1': {'lat_min': -10, 'lat_max': 10, 'lon_min': 50, 'lon_max': 70}, # West
        'dmi_2': {'lat_min': -10, 'lat_max': 0, 'lon_min': 90, 'lon_max': 110}, # East
        'sji850': {'lat_min': 0, 'lat_max': 15, 'lon_min': 35, 'lon_max': 50},
        'sji200': {'lat_min': 0, 'lat_max': 15, 'lon_min': 35, 'lon_max': 50},
        'ueq850': {'lat_min': -4, 'lat_max': 4, 'lon_min': 60, 'lon_max': 90},
        'ueq200': {'lat_min': -4, 'lat_max': 4, 'lon_min': 60, 'lon_max': 90},
        'wp' : {'lat_min': -15, 'lat_max': 20, 'lon_min': 120, 'lon_max': 160},
        'wnp_1' : {'lat_min': 20, 'lat_max': 35, 'lon_min': 160, 'lon_max': 180},
        'wnp_2' : {'lat_min': 20, 'lat_max': 35, 'lon_min': -180, 'lon_max': -150},
        'wsp_1' : {'lat_min': -30, 'lat_max': -15, 'lon_min': 155, 'lon_max': 180}, 
        'wsp_2' : {'lat_min': -30, 'lat_max': -15, 'lon_min': -180, 'lon_max': -150},
    }
    # Get index for region
    return indices_definitions[region]


def standardize_index(data, index_name, period_clm, year_fcst, month_init, before = False):
    # Check if trying to calculate for first month and year
    first_year = data['year'].min().item()
    if year_fcst == first_year and month_init == 1:
        raise ValueError("Cannot calculate the index for the first month in the first year of the dataset.")
    
    # Calculate the index for the entire dataset
    if index_name in ["n4", "dmi", "wnp", "wsp"]: 
        region_1 = get_region_indices(f"{index_name}_1")
        region_2 = get_region_indices(f"{index_name}_2")
        subset_1 = data.sel(lat=slice(region_1['lat_min'], region_1['lat_max']), lon=slice(region_1['lon_min'], region_1['lon_max']))
        subset_2 = data.sel(lat=slice(region_2['lat_min'], region_2['lat_max']), lon=slice(region_2['lon_min'], region_2['lon_max']))
        
        if index_name == "dmi":
            index_dmi_1 = subset_1.mean(dim=['lat', 'lon'])
            index_dmi_2 = subset_2.mean(dim=['lat', 'lon'])
            
            if before:
                # Select the reference period from the indices
                ref_data_dmi_1 = index_dmi_1.sel(year=slice(period_clm[0], period_clm[1]))
                ref_data_dmi_2 = index_dmi_2.sel(year=slice(period_clm[0], period_clm[1]))

                # Calculate the climatology (mean) and standard deviation during the reference period for both indices
                climatology_dmi_1 = ref_data_dmi_1.mean(dim='year')
                climatology_std_dmi_1 = ref_data_dmi_1.std(dim='year', ddof=1)
                climatology_dmi_2 = ref_data_dmi_2.mean(dim='year')
                climatology_std_dmi_2 = ref_data_dmi_2.std(dim='year', ddof=1)

                # Standardize the entire index data
                standardized_index_dmi_1 = (index_dmi_1 - climatology_dmi_1) / climatology_std_dmi_1
                standardized_index_dmi_2 = (index_dmi_2 - climatology_dmi_2) / climatology_std_dmi_2

                # Calculate the difference between the standardized indices
                index = standardized_index_dmi_1 - standardized_index_dmi_2
            else:
                # Calculate the difference between the indices
                index = index_dmi_1 - index_dmi_2
        else:
            combined_subset = xr.concat([subset_1, subset_2], dim='lat')
            index = combined_subset.mean(dim=['lat', 'lon'])

    elif index_name == "wpg":
        # Calculate the index for the entire dataset
        region_wp = get_region_indices(index_name)
        region_n4_1 = get_region_indices("n4_1")
        region_n4_2 = get_region_indices("n4_2")
        subset_wp = data.sel(lat=slice(region_wp['lat_min'], region_wp['lat_max']), lon=slice(region_wp['lon_min'], region_wp['lon_max']))
        subset_n4_1 = data.sel(lat=slice(region_n4_1['lat_min'], region_n4_1['lat_max']), lon=slice(region_n4_1['lon_min'], region_n4_1['lon_max']))
        subset_n4_2 = data.sel(lat=slice(region_n4_2['lat_min'], region_n4_2['lat_max']), lon=slice(region_n4_2['lon_min'], region_n4_2['lon_max']))
        combined_subset_n4 = xr.concat([subset_n4_1, subset_n4_2], dim='lat')
        
        index_wp = subset_wp.mean(dim=['lat', 'lon'])
        index_n4 = combined_subset_n4.mean(dim=['lat', 'lon'])

        if before:
            # Select the reference period from the indices
            ref_data_wp = index_wp.sel(year=slice(period_clm[0], period_clm[1]))
            ref_data_n4 = index_n4.sel(year=slice(period_clm[0], period_clm[1]))

            # Calculate the climatology (mean) and standard deviation during the reference period for both indices
            climatology_wp = ref_data_wp.mean(dim='year')
            climatology_std_wp = ref_data_wp.std(dim='year', ddof=1)
            climatology_n4 = ref_data_n4.mean(dim='year')
            climatology_std_n4 = ref_data_n4.std(dim='year', ddof=1)

            # Standardize the entire index data
            standardized_index_wp = (index_wp - climatology_wp) / climatology_std_wp
            standardized_index_n4 = (index_n4 - climatology_n4) / climatology_std_n4

            # Calculate the difference between the standardized indices
            index = standardized_index_n4 - standardized_index_wp
        else:
            # Calculate the difference between the indices
            index = index_n4 - index_wp

    else:
        region = get_region_indices(index_name)
        subset = data.sel(lat=slice(region['lat_min'], region['lat_max']), lon=slice(region['lon_min'], region['lon_max']))
        index = subset.mean(dim=['lat', 'lon'])

    if before:
        if month_init == 1:
            current_datapoint = index.sel(year=year_fcst-1, month=12)
        else:
            current_datapoint = index.sel(year=year_fcst, month=month_init-1)

        return current_datapoint
    
    else:
        # Select the reference period from the index
        ref_data = index.sel(year=slice(period_clm[0], period_clm[1]))

        # Calculate the climatology (mean) and standard deviation during the reference period
        climatology = ref_data.mean(dim=['year'])
        climatology_std = ref_data.std(dim=['year'], ddof = 1)

        # Standardize the current data point based on data from previous month
        if month_init == 1:
            current_datapoint = index.sel(year = year_fcst-1, month = 12)
            climatology = climatology.sel(month = 12)
            climatology_std = climatology_std.sel(month = 12)
        else:
            current_datapoint = index.sel(year = year_fcst, month = month_init-1)
            climatology = climatology.sel(month = month_init-1)
            climatology_std = climatology_std.sel(month = month_init-1)
        
        anomalies = current_datapoint - climatology
        standardized_anomalies = anomalies / climatology_std
        
        return standardized_anomalies


def standardize_index_diff1(data, index_name, period_clm, year_fcst, month_init, before = False):
    # Check if trying to calculate for first month and year
    first_year = data['year'].min().item()
    if year_fcst == first_year and month_init == 1:
        raise ValueError("Cannot calculate the index for the first month in the first year of the dataset.")
    
    # Calculate the index for the entire dataset
    if index_name == "n34":
        region = get_region_indices(index_name)
        subset = data.sel(lat=slice(region['lat_min'], region['lat_max']), lon=slice(region['lon_min'], region['lon_max']))
        index = subset.mean(dim=['lat', 'lon'])
    elif index_name == "dmi":
        region_dmi_1 = get_region_indices("dmi_1")
        region_dmi_2 = get_region_indices("dmi_2")
        subset_dmi_1 = data.sel(lat=slice(region_dmi_1['lat_min'], region_dmi_1['lat_max']), lon=slice(region_dmi_1['lon_min'], region_dmi_1['lon_max']))
        subset_dmi_2 = data.sel(lat=slice(region_dmi_2['lat_min'], region_dmi_2['lat_max']), lon=slice(region_dmi_2['lon_min'], region_dmi_2['lon_max']))
        index_dmi_1 = subset_dmi_1.mean(dim=['lat', 'lon'])
        index_dmi_2 = subset_dmi_2.mean(dim=['lat', 'lon'])

        if before:
                # Select the reference period from the indices
                ref_data_dmi_1 = index_dmi_1.sel(year=slice(period_clm[0], period_clm[1]))
                ref_data_dmi_2 = index_dmi_2.sel(year=slice(period_clm[0], period_clm[1]))

                # Calculate the climatology (mean) and standard deviation during the reference period for both indices
                climatology_dmi_1 = ref_data_dmi_1.mean(dim='year')
                climatology_std_dmi_1 = ref_data_dmi_1.std(dim='year', ddof=1)
                climatology_dmi_2 = ref_data_dmi_2.mean(dim='year')
                climatology_std_dmi_2 = ref_data_dmi_2.std(dim='year', ddof=1)

                # Standardize the entire index data
                standardized_index_dmi_1 = (index_dmi_1 - climatology_dmi_1) / climatology_std_dmi_1
                standardized_index_dmi_2 = (index_dmi_2 - climatology_dmi_2) / climatology_std_dmi_2

                # Calculate the difference between the standardized indices
                index = standardized_index_dmi_1 - standardized_index_dmi_2

        else:
            # Calculate the difference between the indices
            index = index_dmi_1 - index_dmi_2
    else:
        print(f"Diff1 not implemented for index {index_name}")
        raise TypeError(f"Diff1 not implemented for index {index_name}")

    # Select the reference period from the index
    ref_data = index.sel(year=slice(period_clm[0], period_clm[1]))

    if before:
        # Calculate the climatology (mean) and standard deviation during the reference period
        climatology = ref_data.mean(dim='year')
        climatology_std = ref_data.std(dim='year', ddof=1)

        # Standardize the entire index data
        standardized_index = (index - climatology) / climatology_std

        # Calculate the differences for all 12 months, including January using December from the previous year
        diff_list = []
        for year in range(period_clm[0], period_clm[1] + 1):
            for month in range(1, 13):
                if month == 1:
                    if year > period_clm[0]:  # Ensure we have the previous year's December data
                        current_value = standardized_index.sel(year=year, month=1)
                        previous_value = standardized_index.sel(year=year-1, month=12)
                        diff = (current_value - previous_value).assign_coords(year=year, month=month)
                        diff_list.append(diff)
                else:
                    current_value = standardized_index.sel(year=year, month=month)
                    previous_value = standardized_index.sel(year=year, month=month-1)
                    diff = (current_value - previous_value).assign_coords(year=year, month=month)
                    diff_list.append(diff)

    else:
        # Calculate the differences for all 12 months, including January using December from the previous year
        diff_list = []
        
        for year in range(period_clm[0], period_clm[1] + 1):
            for month in range(1, 13):
                if month == 1:
                    if year > period_clm[0]:  # Ensure we have the previous year's December data
                        current_value = ref_data.sel(year=year, month=1)
                        previous_value = ref_data.sel(year=year-1, month=12)
                        diff = (current_value - previous_value).assign_coords(year=year, month=month)
                        diff_list.append(diff)
                else:
                    current_value = ref_data.sel(year=year, month=month)
                    previous_value = ref_data.sel(year=year, month=month-1)
                    diff = (current_value - previous_value).assign_coords(year=year, month=month)
                    diff_list.append(diff)
    
    diff_data = xr.concat(diff_list, dim='time')

    climatology = diff_data.mean(dim='time')
    climatology_std = diff_data.std(dim='time', ddof = 1)

    if before:
        # Standardize the current data point based on data from the previous month
        if month_init == 1:
            current_datapoint = standardized_index.sel(year = year_fcst, month = 1)
            previous_datapoint = standardized_index.sel(year = year_fcst-1, month = 12)
        else:
            current_datapoint = standardized_index.sel(year = year_fcst, month = month_init)
            previous_datapoint = standardized_index.sel(year = year_fcst, month = month_init-1)
    else:
        # Standardize the current data point based on data from the previous month
        if month_init == 1:
            current_datapoint = index.sel(year = year_fcst, month = 1)
            previous_datapoint = index.sel(year = year_fcst-1, month = 12)
        else:
            current_datapoint = index.sel(year = year_fcst, month = month_init)
            previous_datapoint = index.sel(year = year_fcst, month = month_init-1)

    # Calculate the difference between the current and previous month
    difference = current_datapoint - previous_datapoint

    anomalies = difference - climatology
    standardized_anomalies = anomalies / climatology_std
    
    return standardized_anomalies


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









