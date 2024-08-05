"""
This file contains code for loading and saving data for the lasso forecast.
Raw data on precipitation and from ERA5 can be loaded using load_raw_data.
"""
import numpy as np
import xarray as xr

# Predictors: uwind200, uwind850, sst, chirps
# Dir: era5_dir or chirps_dir
 
def load_raw_data(dir, predictor, years, season, lat_bnds = None, lon_bnds = None):
    """
    Load and preprocess raw data for a given predictor within specified bounds and time period.

    Parameters:
    - dir (str): The directory path where the raw data files are stored. This could be `era5_dir` or `chirps_dir`.
    - predictor (str): The predictor to load. Valid options are "sst", "uwind200", "uwind850", and "chirps".
    - years (list): A list of years (integers) to subset the data.
    - season (str, optional): The season to aggregate the data by. Valid options are 'OND', 'MAM', 'JJAS'.
    - lat_bnds (list): A list of two floats representing the geographical bounding of the latitude (lat_min, lat_max).
    - lon_bnds (list): A list of two floats representing the geographical bounding of the longitude (lon_min, lon_max).

    Returns:
    - If predictor is "chirps":
      - year (numpy.ndarray): 1D array of years.
      - lat (numpy.ndarray): 1D array of latitudes.
      - lon (numpy.ndarray): 1D array of longitudes.
      - predictor_values (numpy.ndarray): 3D array of predictor values with shape (year, lat, lon).

    - If predictor is not "chirps":
      - data (xarray.Dataset): The loaded xarray Dataset object containing the data for the specified predictor.
    """
    # Construct file path
    if predictor == "sst":
        pth = "sst/full.nc"
    elif predictor == "uwind200":
        pth = "uwind/200hpa/full.nc"
    elif predictor == "uwind850":
        pth = "uwind/850hpa/full.nc"
    elif predictor == "chirps":
        pth = "prec_raw.nc"
    file_path = f"{dir}{pth}"
    
    # Load data
    data = xr.open_dataset(file_path)
    
    if predictor == "chirps":
        # Subset data based on lat_bnds and lon_bnds
        data = data.sel(lat=slice(lat_bnds[0], lat_bnds[1]), lon=slice(lon_bnds[0], lon_bnds[1]))
    
    # Subset data based on years
    data = data.sel(year=slice(years[0], years[-1]))
    
    # Aggregate by season for chirps data
    if predictor == "chirps":
        if season == 'OND':
            season_months = [10, 11, 12]
        elif season == 'MAM':
            season_months = [3, 4, 5]
        elif season == 'JJAS':
            season_months = [6, 7, 8, 9]
        data = data.sel(month=np.isin(data['month'], season_months))
        
        # Aggregate by summing over the selected months
        data = data.groupby('year').sum(dim='month', skipna=False)

        # Convert to NumPy arrays
        year = data['year'].values
        lat = data['lat'].values
        lon = data['lon'].values
        predictor_values = data['prec'].values # Shape: (year, lat, lon)
        return year, lat, lon, predictor_values
    else:
        return data


def save_anomalies(anomalies, year, lat, lon, dir, season, normalized=False):
    """
    Save the calculated anomalies to a NetCDF file.

    Parameters:
    - anomalies (numpy.ndarray): A 3D numpy array of anomalies with dimensions (year, lat, lon).
    - year (numpy.ndarray): A 1D numpy array of years corresponding to the first dimension of `anomalies`.
    - lat (numpy.ndarray): A 1D numpy array of latitudes corresponding to the second dimension of `anomalies`.
    - lon (numpy.ndarray): A 1D numpy array of longitudes corresponding to the third dimension of `anomalies`.
    - dir (str): The directory path where the NetCDF file will be saved.
    - season (str): The season to save anomalies for ('MAM', 'JJAS' or 'OND').
    - normalized (bool, optional): A flag indicating whether the anomalies are normalized. Default is False.

    Returns:
    - None: The function saves the anomalies to a NetCDF file in the specified directory.
    """
    # Convert to xarray DataArray for saving as NetCDF
    anomalies_xr = xr.DataArray(anomalies, coords=[year, lat, lon], dims=['year', 'lat', 'lon'])

    # Save the anomalies
    if normalized:
        print("Saving normalized anomalies...")
        anomalies_xr.to_netcdf(f"{dir}chirps_anomalies_normal_{season}.nc")
        print(f"Normalized anomalies saved to: {dir}chirps_anomalies_{season}_normal.nc")
    else:
        print("Saving anomalies...")
        anomalies_xr.to_netcdf(f"{dir}chirps_anomalies_{season}.nc")
        print(f"Anomalies saved to: {dir}chirps_anomalies_{season}.nc")


def save_eofs_pcs(eofs_reshaped, pcs, var_fractions, year, lat, lon, dir, season):
    """
    Save the calculated EOFs, PCs, and variance fractions to NetCDF files.

    Parameters:
    - eofs_reshaped (numpy.ndarray): A 3D numpy array of EOFs with dimensions (n_eofs, lat, lon).
    - pcs (numpy.ndarray): A 2D numpy array of PCs with dimensions (year, n_eofs).
    - var_fractions (numpy.ndarray): A 1D numpy array of the fraction of variance explained by each EOF.
    - year (numpy.ndarray): A 1D numpy array of years corresponding to the first dimension of `pcs`.
    - lat (numpy.ndarray): A 1D numpy array of latitudes corresponding to the spatial dimensions of `eofs`.
    - lon (numpy.ndarray): A 1D numpy array of longitudes corresponding to the spatial dimensions of `eofs`.
    - dir (str): The directory path where the NetCDF files will be saved.
    - season (str): The season to save anomalies for ('MAM', 'JJAS' or 'OND').

    Returns:
    - None: The function saves the EOFs, PCs, and variance fractions to NetCDF files in the specified directory.
    """

    # Convert EOFs, PCs and variance fractions to xarray DataArrays for saving as NetCDFs
    eofs_xr = xr.DataArray(eofs_reshaped, coords=[range(eofs_reshaped.shape[0]), lat, lon], dims=['eof', 'lat', 'lon'])
    pcs_xr = xr.DataArray(pcs, coords=[year, range(pcs.shape[1])], dims=['year', 'eof'])
    var_fractions_xr = xr.DataArray(var_fractions, coords=[range(var_fractions.shape[0])], dims=['eof'])

    # Save the EOFs
    eofs_file_path = f"{dir}chirps_eofs_{season}.nc"
    print(f"Saving EOFs...")
    eofs_xr.to_netcdf(eofs_file_path)
    print(f"EOFs saved to: {eofs_file_path}")

    # Save the PCs
    pcs_file_path = f"{dir}chirps_pcs_{season}.nc"
    print(f"Saving PCs...")
    pcs_xr.to_netcdf(pcs_file_path)
    print(f"PCs saved to: {pcs_file_path}")

    # Save the variance fractions
    var_fractions_file_path = f"{dir}chirps_var_fracs_{season}.nc"
    print(f"Saving variance fractions...")
    var_fractions_xr.to_netcdf(var_fractions_file_path)
    print(f"Variance fractions saved to: {var_fractions_file_path}")


def save_model_results(df_coefficients, df_fl_pred_cov, dir, season, month_init, n_eofs):
    """
    Save the model coefficients and prediction covariances to NetCDF files.

    Parameters:
    - df_coefficients (pd.DataFrame): A DataFrame containing the model coefficients.
    - df_fl_pred_cov (pd.DataFrame): A DataFrame containing the prediction covariances.
    - dir (str): The directory path where the NetCDF files will be saved.
    - season (str): The season to save anomalies for ('MAM', 'JJAS' or 'OND').
    - n_eofs (int): Number of EOFs (default is 7).

    Returns:
    - None: The function saves the model results to NetCDF files in the specified directory.
    """

    # Convert coefficients DataFrame to xarray Dataset
    coefficients_xr = xr.Dataset.from_dataframe(df_coefficients)

    # Save the coefficients
    coefficients_file_path = f"{dir}model_coefficients_{season}_month_{month_init}.nc"
    print(f"Saving model coefficients...")
    coefficients_xr.to_netcdf(coefficients_file_path)
    print(f"Model coefficients saved to: {coefficients_file_path}")

    # Reshape the prediction covariances
    years = df_fl_pred_cov.index
    cov_array = np.array([df_fl_pred_cov.loc[year].values.reshape(n_eofs, n_eofs) for year in years])
    
    # Create an xarray Dataset for covariances
    cov_ds = xr.Dataset(
        data_vars={'covariance': (['year', 'eof_i', 'eof_j'], cov_array)},
        coords={'year': years, 'eof_i': range(1, n_eofs+1), 'eof_j': range(1, n_eofs+1)}
    )

    # Save the prediction covariances
    covariances_file_path = f"{dir}prediction_covariances_{season}_month_{month_init}.nc"
    print(f"Saving prediction covariances...")
    cov_ds.to_netcdf(covariances_file_path)
    print(f"Prediction covariances saved to: {covariances_file_path}")

