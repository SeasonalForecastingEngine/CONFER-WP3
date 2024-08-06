"""
This file contains code for technical calculations used in the lasso forecast.
"""
import numpy as np
import pandas as pd
import xarray as xr

from eofs.standard import Eof
from functools import reduce
from scipy.stats  import norm, pearsonr
from scipy.interpolate import interp1d
from sklearn.linear_model import MultiTaskLassoCV


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

    # Code for flipping value of first EOF if it is negative.
    # This helps with the interpretability of the plot of model coefficients, as the coefficients for eof1 will correspond with a positive sign in the plot.
    # Check if the first EOF is predominantly negative
    if np.sum(eofs[0] < 0) > np.sum(eofs[0] > 0):
        # If it is, flip the sign of the first EOF and its corresponding PC
        eofs[0] *= -1
        pcs[:, 0] *= -1
    
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


# Helper functions for standardize_index and standardize_index_diff1
def get_subset(data, region_name):
    """
    Extracts a subset of the data for a specified region.

    Args:
    data (xarray.Dataset): The dataset containing the data to be subset.
    region_name (str): The name of the region to extract.

    Returns:
    xarray.Dataset: A subset of the original dataset for the specified region.
    """
    region = get_region_indices(region_name)
    return data.sel(lat=slice(region['lat_min'], region['lat_max']), 
                    lon=slice(region['lon_min'], region['lon_max']))


def calc_index(subset):
    """
    Calculates the mean value of the data subset over the latitude and longitude dimensions.

    Args:
    subset (xarray.Dataset): The subset of data to calculate the index for.

    Returns:
    xarray.DataArray: The mean value of the subset over the latitude and longitude dimensions.
    """
    return subset.mean(dim=['lat', 'lon'])
    

def standardize_index(data, index_name, period_clm, year_fcst, month_init):
    """
    Calculates and standardizes an index for a given dataset, index name, reference period, forecast year, and initialization month.

    Args:
    data (xarray.Dataset): The dataset containing the data to be used for index calculation.
    index_name (str): The name of the index to be calculated (e.g., 'n4', 'wnp'. See get_region_indices() for the possible indices.).
    period_clm (tuple): A tuple containing the start and end years (inclusive) for the reference period.
    year_fcst (int): The forecast year for which the index is being standardized.
    month_init (int): The initialization month for the forecast.

    Returns:
    xarray.DataArray: The standardized index value for the specified forecast year and initialization month.
    """
    # Calculate the index
    if index_name in ["n4", "wnp", "wsp"]:
        # Indices calculated from two regions
        subset_1 = get_subset(data, f"{index_name}_1")
        subset_2 = get_subset(data, f"{index_name}_2")
        index = calc_index(xr.concat([subset_1, subset_2], dim='lat'))
    elif index_name in ["dmi", "wpg"]:
        # Indices that are a difference between regions
        if index_name == "dmi":
            # Swapped to get correct index at the end
            subset_2 = get_subset(data, f"dmi_1")
            subset_1 = get_subset(data, f"dmi_2")
        else:
            subset_1 = get_subset(data, index_name)
            subset_n4_1 = get_subset(data, "n4_1")
            subset_n4_2 = get_subset(data, "n4_2")
            subset_2 = xr.concat([subset_n4_1, subset_n4_2], dim='lat')
            
        index_1 = calc_index(subset_1)
        index_2 = calc_index(subset_2)

        # Calculate the climatology (mean) and standard deviation during the reference period for both indices
        climatology_1 = index_1.sel(year=slice(*period_clm)).mean(dim='year')
        climatology_std_1 = index_1.sel(year=slice(*period_clm)).std(dim='year', ddof=1)
        climatology_2 = index_2.sel(year=slice(*period_clm)).mean(dim='year')
        climatology_std_2 = index_2.sel(year=slice(*period_clm)).std(dim='year', ddof=1)

        # Standardize the entire index data
        standardized_index_1 = (index_1 - climatology_1) / climatology_std_1
        standardized_index_2 = (index_2 - climatology_2) / climatology_std_2

        # Calculate the difference between the standardized indices
        index = standardized_index_2 - standardized_index_1
    else:
        # Indices calculated from one region
        index = calc_index(get_subset(data, index_name))

    # Get the appropriate datapoint
    if month_init == 1:
        current_datapoint = index.sel(year=year_fcst-1, month=12)
    else:
        current_datapoint = index.sel(year=year_fcst, month=month_init-1)

    if index_name in ["dmi", "wpg"]:
        return current_datapoint
    
    # Select the reference period from the index
    ref_data = index.sel(year=slice(*period_clm))

    # Calculate the climatology (mean) and standard deviation during the reference period
    climatology = ref_data.mean('year').sel(month=current_datapoint.month)
    climatology_std = ref_data.std('year').sel(month=current_datapoint.month)
    
    return (current_datapoint - climatology) / climatology_std


def standardize_index_diff1(data, index_name, period_clm, year_fcst, month_init):
    """
    Calculates the difference between the current and previous month's index values for a given dataset, index name, reference period, forecast year, and initialization month.

    Args:
    data (xarray.Dataset): The dataset containing the data to be used for index calculation.
    index_name (str): The name of the index to be calculated ('n34' or 'dmi').
    period_clm (tuple): A tuple containing the start and end years (inclusive) for the reference period.
    year_fcst (int): The forecast year for which the index difference is being calculated.
    month_init (int): The initialization month for the forecast.

    Returns:
    xarray.DataArray: The standardized difference between the current and previous month's index values for the specified forecast year and initialization month.
    """
    # Calculate the index
    if index_name == "n34":
        index = calc_index(get_subset(data, index_name))
        # Calculate the climatology (mean) and standard deviation during the reference period
        climatology = index.sel(year=slice(*period_clm)).mean(dim='year')
        climatology_std = index.sel(year=slice(*period_clm)).std(dim='year', ddof=1)
        # Standardize the entire index data
        index = (index - climatology) / climatology_std
    elif index_name == "dmi":
        index_1 = calc_index(get_subset(data, "dmi_1"))
        index_2 = calc_index(get_subset(data, "dmi_2"))
        # Calculate the climatology (mean) and standard deviation during the reference period for both indices
        climatology_1 = index_1.sel(year=slice(*period_clm)).mean(dim='year')
        climatology_std_1 = index_1.sel(year=slice(*period_clm)).std(dim='year', ddof=1)
        climatology_2 = index_2.sel(year=slice(*period_clm)).mean(dim='year')
        climatology_std_2 = index_2.sel(year=slice(*period_clm)).std(dim='year', ddof=1)
        # Standardize the entire index data
        index_1 = (index_1 - climatology_1) / climatology_std_1
        index_2 = (index_2 - climatology_2) / climatology_std_2
        # Calculate the difference between the indices
        index = index_1 - index_2
    else:
        raise TypeError(f"Diff1 not implemented for index {index_name}")

    # Calculate the difference between the current and previous month
    if month_init == 1:
        current = index.sel(year=year_fcst-1, month=12)
        previous = index.sel(year=year_fcst-1, month=11)
    elif month_init == 2:
        current = index.sel(year=year_fcst, month=1)
        previous = index.sel(year=year_fcst-1, month=12)
    else:
        current = index.sel(year=year_fcst, month=month_init-1)
        previous = index.sel(year=year_fcst, month=month_init-2)
    
    difference = current - previous

    return difference

    # Code that standardizes one more time after calculating differences.
    # This is an alternative implementation that could also be sensible, but is not currently used in order to replicate previously calculated indices.
    # Would need some updating to work.
"""
    # Calculate climatology and standard deviation of differences
    diff_data = xr.concat([
        ref_data.diff('month').assign_coords(month=range(2, 13)),
        (ref_data.shift(year=-1).isel(month=0) - ref_data.isel(month=-1)).assign_coords(month=1)
    ], dim='month')
    
    climatology = diff_data.mean('year')
    climatology_std = diff_data.std('year')

    # Calculate anomalies and standardize
    anomalies = difference - climatology.sel(month=month_init)
    standardized_anomalies = anomalies / climatology_std.sel(month=month_init)
    
    return standardized_anomalies
"""


def prepare_time_series_data(data, index_name, period_clm, period_train, months, diff1=False):
    """
    Prepares the time series data for a given index over a specified training period.

    Args:
    data (xarray.Dataset): The dataset containing the data to be used for index calculation.
    index_name (str): The name of the index to be calculated.
    period_clm (tuple): A tuple containing the start and end years (inclusive) for the reference period.
    period_train (tuple): A tuple containing the start and end years (inclusive) for the training period.
    months (list): A list of months to calculate the index for.
    diff1 (bool, optional): Flag indicating whether to calculate the first difference of the index. Default is False.

    Returns:
    pd.DataFrame: A DataFrame containing the year, month, and standardized anomaly for the specified index.
    """
    time_series_data = []

    for year in range(period_train[0], period_train[1] + 1):
        for month in months:
            # Correct for functions calculating value for month-1
            if month == 12:
                year_corrected = year + 1
                month_corrected = 1
            else:
                year_corrected = year
                month_corrected = month + 1
            
            # Set variable fixing the problem of not having data to compute first value
            first_val = False

            if diff1:
                # Deal with the first datapoint not available by setting the anomaly to 0
                if year == period_train[0] and month == 1:
                    first_val = True    # Sets to 0 further down in the function
                else:
                    standardized_anomaly = standardize_index_diff1(data, index_name, period_clm, year_corrected, month_corrected)
                # Alternatively, deal with first datapoint not available by copying the first available one
                # if year == period_train[0] and month == 1:
                #     month_corrected += 1
                #     standardized_anomaly = standardize_index_diff1(data, index_name, period_clm, year_corrected, month_corrected)
            else:
                standardized_anomaly = standardize_index(data, index_name, period_clm, year_corrected, month_corrected)
                
            if index_name in ["ueq850", "ueq200", "sji850", "sji200"]:
                standardized_anomaly = standardized_anomaly.uwind.values
            elif first_val:
                standardized_anomaly = 0.
            else:
                standardized_anomaly = standardized_anomaly.sst.values

            time_series_data.append({
                'year': year,
                'month': month,
                'standardized_anomaly': standardized_anomaly
            })

    df = pd.DataFrame(time_series_data)

    # Ensure the 'year' and 'month' columns are integers
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    
    # Ensure the DataFrame has the correct format with 'year', 'month', and 'standardized_anomaly' as columns
    df = df[['year', 'month', 'standardized_anomaly']]
    
    return df


def get_all_indices(sst_data, uwind200_data, uwind850_data, period_clm, period_train, months):
    """
    Calculates multiple indices and prepares their time series data for a specified training period.

    Args:
    sst_data (xarray.Dataset): The dataset containing sea surface temperature (SST) data.
    uwind200_data (xarray.Dataset): The dataset containing 200 hPa zonal wind data.
    uwind850_data (xarray.Dataset): The dataset containing 850 hPa zonal wind data.
    period_clm (tuple): A tuple containing the start and end years (inclusive) for the reference period.
    period_train (tuple): A tuple containing the start and end years (inclusive) for the training period.
    months (list): A list of months to calculate the indices for.

    Returns:
    pd.DataFrame: A DataFrame containing the year, month, and standardized anomalies for all specified indices.
    """
    index_names = ['n34', 'n3', 'n4', 'dmi', 'n34_diff1', 'dmi_diff1', 'wsp', 'wpg', 'wp', 'wnp', 'ueq850', 'ueq200', 'sji850', 'sji200']
    index_dfs = {}

    for index in index_names:
        if index.endswith("diff1"):
            time_series_df = prepare_time_series_data(sst_data, index[:3], period_clm, period_train, months, diff1=True)
        elif index in ['ueq850', 'sji850']:
            time_series_df = prepare_time_series_data(uwind850_data, index, period_clm, period_train, months)
        elif index in ['ueq200', 'sji200']:
            time_series_df = prepare_time_series_data(uwind200_data, index, period_clm, period_train, months)
        else:
            time_series_df = prepare_time_series_data(sst_data, index, period_clm, period_train, months)
        
        index_dfs[index] = time_series_df

    # Add wvg as well
    # Merge DataFrames
    merge_for_wvg_df = reduce(lambda left, right: pd.merge(left, right, on=['year', 'month']), (
        index_dfs["n4"].rename(columns={'standardized_anomaly': 'n4'}),
        index_dfs["wp"].rename(columns={'standardized_anomaly': 'wp'}),
        index_dfs["wnp"].rename(columns={'standardized_anomaly': 'wnp'}),
        index_dfs["wsp"].rename(columns={'standardized_anomaly': 'wsp'}))
    )
    merge_for_wvg_df['wvg'] = merge_for_wvg_df['n4'] - (merge_for_wvg_df['wp'] + merge_for_wvg_df['wnp'] + merge_for_wvg_df['wsp']) / 3
    index_dfs["wvg"] = merge_for_wvg_df[['year', 'month', 'wvg']].rename(columns={'wvg': 'standardized_anomaly'})

    # Combine all DataFrames into a single DataFrame
    era5_indices = reduce(lambda left, right: pd.merge(left, right, on=['year', 'month']), 
                          [df.rename(columns={'standardized_anomaly': index}) for index, df in index_dfs.items()])
    
    return era5_indices


def get_ml_results(era5_indices, feature_names, pcs, var_fracs, n_eofs, period_train, period_clm, month_init):
    """
    Fit a multi-task Lasso regression model and compute the prediction covariance.

    Parameters:
    - era5_indices (pd.DataFrame): DataFrame containing ERA5 indices with columns ['year', 'month'].
    - feature_names (list): List of feature names to be used in the regression model.
    - pcs (numpy.ndarray): Principal components as a 2D array with dimensions (years, n_eofs).
    - var_fracs (numpy.ndarray): Array containing the variance fractions explained by each EOF.
    - n_eofs (int): Number of EOFs to use.
    - period_train (tuple): Tuple (start_year, end_year) defining the training period.
    - period_clm (tuple): Tuple (start_year, end_year) defining the climatology period.
    - month_init (int): The initial month for the model predictions.

    Returns:
    - df_coefficients (pd.DataFrame): DataFrame containing the fitted model coefficients.
    - df_fl_pred_cov (pd.DataFrame): DataFrame containing the prediction covariance matrix for each year.
    """
    # Cross-validation setup
    years_cv = list(range(period_train[0], period_train[1] + 1))  # Training years
    years_verif = list(range(period_clm[0], period_clm[1] + 1))   # Verification years

    # Select the data for the month before month_init
    previous_month = month_init - 1 if month_init > 1 else 12
    df_combined_features = era5_indices[(era5_indices['year'].isin(years_cv)) & (era5_indices['month'] == previous_month)].set_index('year')[feature_names]
    
    # Ensure there are no missing values
    df_combined_features.fillna(0, inplace=True)
    
    # Create target DataFrame for the principal components (PCs)
    df_target = pd.DataFrame(pcs[:, :n_eofs], index=years_cv).reindex(years_cv)
    y = df_target.to_numpy()
    X = df_combined_features.to_numpy()

    # Feature pre-selection
    wgt = np.sqrt(var_fracs / np.sum(var_fracs))  # Calculate weights
    feature_idx = [True] + [False] * len(df_combined_features.columns)
    for ift in range(len(feature_idx) - 1):
        pval = [pearsonr(y[:, ipc], X[:, ift])[1] for ipc in range(n_eofs)]
        feature_idx[1 + ift] = np.any(np.array(pval) < 0.1 * wgt)

    # Filter the selected features
    df_combined_features = df_combined_features.iloc[:, feature_idx[1:]]
    df_year = pd.DataFrame((df_combined_features.index - 2000) / 10, index=df_combined_features.index, columns=['year'])
    df_combined_features = pd.concat([df_year, df_combined_features], axis=1)
    
    X = df_combined_features.to_numpy()

    # Cross-validation folds
    k = 5
    cv_folds = []
    for i in range(k):
        idx_test = set(range(i * 2, len(years_cv), k * 2)).union(set(range(1 + i * 2, len(years_cv), k * 2)))
        idx_train = set(range(len(years_cv))) - idx_test
        cv_folds.append((list(idx_train), list(idx_test)))

    # Lasso regression
    clf = MultiTaskLassoCV(cv=cv_folds, fit_intercept=False, max_iter=5000)
    clf.fit(X, y)

    # Make coefficient DataFrame
    df_coefficients = pd.DataFrame(clf.coef_, index=[f'eof{i}' for i in range(1, n_eofs + 1)], columns=df_combined_features.columns)

    # Compute prediction covariance
    df_fl_pred_cov = pd.DataFrame(index=years_verif, columns=[f'cov-{i}{j}' for i in range(1, n_eofs + 1) for j in range(1, n_eofs + 1)])  # DataFrame for storing results
    errors = y - clf.predict(X)
    ind_active = np.all(clf.coef_ != 0, axis=0)
    n_a = sum(ind_active)
    dgf = 1 + n_a if n_a > 0 else 1
    prediction_cov = np.dot(errors.T, errors) / (len(years_cv) - dgf)
    df_fl_pred_cov.iloc[:, :] = np.broadcast_to(prediction_cov.flatten(), (len(years_verif), n_eofs ** 2))
    
    return df_coefficients, df_fl_pred_cov


def calculate_tercile_probability_forecasts(era5_indices, anomalies_normal, eofs_reshaped, df_coefficients, df_fl_pred_cov, var_fracs, feature_names, year, period_clm, n_eofs, year_fcst, month_init):
    """
    Calculate tercile probability forecasts for precipitation amounts using machine learning model coefficients and EOFs.
    This function calculates the below-normal (prob_bn) and above-normal (prob_an) tercile probabilities
    for precipitation amounts based on machine learning model predictions.

    The plot of combined tercile probabilities currently never displays values in the normal category. 
    I am not sure why, but I think this is most likely due to the values of prob_bn and prob_an coming from this function always being too high to let normal become the most likely category.

    Parameters:
    - era5_indices (pd.DataFrame): DataFrame containing ERA5 indices with columns ['year', 'month'] and other features.
    - anomalies_normal (numpy.ndarray): 3D array of normalized anomalies with dimensions (time, lat, lon).
    - eofs_reshaped (numpy.ndarray): 3D array of EOFs reshaped with dimensions (n_eofs, lat, lon).
    - df_coefficients (pd.DataFrame): DataFrame containing the model coefficients.
    - df_fl_pred_cov (pd.DataFrame): DataFrame containing the prediction covariance matrix for each year.
    - var_fracs (numpy.ndarray): Array containing the variance fractions explained by each EOF.
    - feature_names (list): List of feature names to be used in the regression model.
    - year (numpy.ndarray): 1D array of years corresponding to the anomalies_normal and era5_indices data.
    - period_clm (tuple): Tuple containing the start and end years of the climatology period (e.g., (1993, 2020)).
    - n_eofs (int): Number of EOFs to use.
    - year_fcst (int): The forecast year for which the probabilities are being calculated.
    - month_init (int): The initialization month for the model predictions.

    Returns:
    - prob_bn (numpy.ndarray): 2D array of probabilities for below-normal precipitation.
    - prob_an (numpy.ndarray): 2D array of probabilities for above-normal precipitation.
    """
    
    # Select the reference period using a boolean mask
    ref_period_mask = (year >= period_clm[0]) & (year <= period_clm[1])
    ref_period_indices = np.where(ref_period_mask)[0]
    
    # Define the previous month
    previous_month = month_init - 1 if month_init > 1 else 12
    years_verif = list(range(period_clm[0], period_clm[1] + 1))

    # Reshape the covariance matrix for each year
    df_fl_pred_cov_reshaped = np.array([df_fl_pred_cov.loc[yr].values.reshape(n_eofs, n_eofs) for yr in df_fl_pred_cov.index])

    # Use the reshaped covariance matrix for the forecast year
    cov_matrix_for_year = df_fl_pred_cov_reshaped[years_verif.index(year_fcst)]

    # Calculate the standard deviation of anomalies for the reference period
    scaling = np.nanstd(anomalies_normal[ref_period_indices, :, :], axis=0, ddof=1)

    # Select the data for the forecast year and previous month
    year_month_mask = (era5_indices['year'] == year_fcst) & (era5_indices['month'] == previous_month)
    ts_indices = era5_indices.loc[year_month_mask, feature_names].iloc[0]

    # Ensure alignment before using the function
    coefficients_columns = df_coefficients.columns.to_list()
    ts_indices = ts_indices.reindex(coefficients_columns)
    
    # Calculate residual variance
    var_eps = scaling**2 - np.sum(var_fracs[:, None, None] * eofs_reshaped**2, axis=0)

    # Ensure ts_indices has year standardized
    ts_indices['year'] = (year_fcst - 2000) / 10

    # Calculate predictive mean of factor loadings
    fl_eof_mean = df_coefficients.dot(ts_indices).to_numpy()

    # Calculate mean and variance of the probabilistic forecast in normal space
    mean_ml = np.array(np.sum(fl_eof_mean[:, None, None] * eofs_reshaped, axis=0), dtype=np.float64)
    var_ml = np.array(np.sum(np.sum(cov_matrix_for_year[:, :, None, None] * eofs_reshaped[None, :, :, :], axis=1) * eofs_reshaped, axis=0) + var_eps, dtype=np.float64)

    # Standardize mean and variance
    mean_ml_stdz = mean_ml / scaling
    stdv_ml_stdz = np.sqrt(var_ml) / scaling

    # Calculate tercile forecasts
    prob_bn = norm.cdf((norm.ppf(0.333) - mean_ml_stdz) / stdv_ml_stdz)
    prob_an = 1.0 - norm.cdf((norm.ppf(0.667) - mean_ml_stdz) / stdv_ml_stdz)

    return prob_bn, prob_an
