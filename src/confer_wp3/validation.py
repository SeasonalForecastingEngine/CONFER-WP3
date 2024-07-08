"""
This file contains code for validating the calculations made in lasso_fcst_example. 
The code is not needed to run the forecast.
"""
import matplotlib.colors as mcolors 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from scipy.stats import norm


def validate_anomalies1(prec_data, anomalies, lat, lon, year_index=0):
    """
    Visualize the original precipitation data and calculated anomalies on a map for a single year.

    Parameters:
    - prec_data (numpy.ndarray): A 3D numpy array representing the precipitation data with dimensions (time, lat, lon).
    - anomalies (numpy.ndarray): A 3D numpy array representing the precipitation anomalies with the same dimensions as `prec_data`.
    - lat (numpy.ndarray): A 1D numpy array representing the latitude values.
    - lon (numpy.ndarray): A 1D numpy array representing the longitude values.
    - year_index (int): Integer indexing the prec_data for a year to be plotted. Set to the first year by default.

    Returns:
    None: This function displays two plots - one for the original precipitation data and one for the anomalies.
    """
    
    # Create the meshgrid for lat and lon
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Plot original precipitation data
    plt.figure(figsize=(8, 10))  # Adjusting figsize for a more even aspect ratio
    plt.subplot(2, 1, 1)
    plt.title('Original Precipitation Data')
    plt.pcolormesh(lon_grid, lat_grid, prec_data[year_index, :, :], cmap='viridis', shading='auto')
    plt.colorbar(label='Precipitation (mm)')
    plt.gca().set_aspect('equal', adjustable='box')

    # Plot anomalies
    plt.subplot(2, 1, 2)
    plt.title('Precipitation Anomalies')
    plt.pcolormesh(lon_grid, lat_grid, anomalies[year_index, :, :], cmap='bwr', shading='auto')
    plt.colorbar(label='Anomalies (mm)')
    plt.gca().set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


def validate_anomalies2(anomalies, anomalies_normal, lat, lon, year_index=0):
    """
    Visualize and compare original and transformed precipitation anomalies for a single year.

    Parameters:
    - anomalies (numpy.ndarray): A 3D numpy array of precipitation anomalies with dimensions (year, lat, lon).
    - anomalies_normal (numpy.ndarray): A 3D numpy array of transformed precipitation anomalies with dimensions (year, lat, lon).
    - lat (numpy.ndarray): A 1D numpy array of latitude values.
    - lon (numpy.ndarray): A 1D numpy array of longitude values.
    - year_index (int, optional): The index of the year to visualize. Default is 0.

    Returns:
    - None: The function generates two side-by-side plots comparing the original and transformed precipitation anomalies for the specified year.
    """

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

    # Plot Precipitation Anomalies
    pltrg = np.nanmax(abs(anomalies[year_index, :, :]))
    axes[0].set_title('Precipitation Anomalies')
    img1 = axes[0].imshow(anomalies[year_index, :, :], origin='lower', 
                        extent=[lon.min(), lon.max(), lat.min(), lat.max()], 
                        cmap='bwr', aspect='auto', vmin=-pltrg, vmax=pltrg)
    fig.colorbar(img1, ax=axes[0], label='Anomalies (mm)')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')

    # Plot Transformed Precipitation Anomalies
    pltrg_normal = np.nanmax(abs(anomalies_normal[year_index, :, :]))
    axes[1].set_title('Transformed Precipitation Anomalies')
    img2 = axes[1].imshow(anomalies_normal[year_index, :, :], origin='lower', 
                        extent=[lon.min(), lon.max(), lat.min(), lat.max()], 
                        cmap='bwr', aspect='auto', vmin=-pltrg_normal, vmax=pltrg_normal)
    fig.colorbar(img2, ax=axes[1], label='Transformed Anomalies (mm)')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')

    plt.tight_layout()
    plt.show()
    

def validate_anomalies3(anomalies, anomalies_normal):
    """
    Visualize and compare the distribution of original and transformed precipitation anomalies using histograms.

    Parameters:
    - anomalies (numpy.ndarray): A 3D numpy array of precipitation anomalies with dimensions (year, lat, lon).
    - anomalies_normal (numpy.ndarray): A 3D numpy array of transformed precipitation anomalies with dimensions (year, lat, lon).

    Returns:
    - None: The function generates two histograms side-by-side comparing the distributions of original and transformed precipitation anomalies.
    """
    
    # Flatten the arrays for plotting histograms
    anomalies_flat = anomalies.flatten()
    anomalies_normal_flat = anomalies_normal.flatten()

    # Remove non-finite values
    anomalies_flat = anomalies_flat[np.isfinite(anomalies_flat)]
    anomalies_normal_flat = anomalies_normal_flat[np.isfinite(anomalies_normal_flat)]

    plt.figure(figsize=(14, 6))

    # Original anomalies histogram
    plt.subplot(1, 2, 1)
    plt.hist(anomalies_flat, bins=30, density=True, alpha=0.6, color='g')
    plt.title('Histogram of Anomalies')
    plt.xlabel('Anomaly Value')
    plt.ylabel('Density')
    plt.axvline(np.mean(anomalies_flat), color='r', linestyle='dashed', linewidth=1)  # Add dashed line of mean
    plt.legend(['Mean'])

    # Transformed anomalies histogram
    plt.subplot(1, 2, 2)
    plt.hist(anomalies_normal_flat, bins=30, density=True, alpha=0.6, color='b')

    # Overlay normal distribution curve
    mu, std = norm.fit(anomalies_normal_flat)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)

    plt.title('Histogram of Transformed Anomalies')
    plt.xlabel('Transformed Anomaly Value')
    plt.ylabel('Density')
    plt.axvline(mu, color='r', linestyle='dashed', linewidth=1)
    plt.legend(['Normal Fit', 'Mean'])

    plt.tight_layout()
    plt.show()


def validate_anomalies4(anomalies, anomalies_normal):
    """
    Print statistical information about the original and transformed precipitation anomalies.

    Parameters:
    - anomalies (numpy.ndarray): A 3D numpy array of precipitation anomalies with dimensions (year, lat, lon).
    - anomalies_normal (numpy.ndarray): A 3D numpy array of transformed precipitation anomalies with dimensions (year, lat, lon).

    Returns:
    - None: The function prints statistical information to the console, including the mean, standard deviation, and the count of NaN and infinite values.
    """
    print("Printing some information about anomalies.")
    print("Anomalies - mean:", np.nanmean(anomalies), "std:", np.nanstd(anomalies))
    print("Transformed Anomalies - mean:", np.nanmean(anomalies_normal), "std:", np.nanstd(anomalies_normal))

    # Check for any Inf values (should be zero)
    print("Inf values in anomalies:", np.isinf(anomalies).sum())
    print("Inf values in transformed anomalies:", np.isinf(anomalies_normal).sum())

    # Check for any NaN values (should be some from mask)
    print("NaN values in anomalies:", np.isnan(anomalies).sum())
    print("NaN values in transformed anomalies:", np.isnan(anomalies_normal).sum())

    
def validate_eofs(eofs_reshaped, title, n_eofs):
    """
    Visualize the Empirical Orthogonal Functions (EOFs) as a series of plots.

    Parameters:
    - eofs (numpy.ndarray): A 2D numpy array of the computed EOFs with dimensions (n_eofs, lat, lon).
    - title (str): The title for the entire plot.
    - n_eofs (int): The number of EOFs to plot.

    Returns:
    - None: The function generates a series of plots for the specified number of EOFs.
    """
    eofs_masked = np.ma.masked_invalid(eofs_reshaped)
    fig, axes = plt.subplots(1, n_eofs, figsize=(20, 5), constrained_layout=True)
    
    # Determine the color scale limits based on the maximum absolute value in the EOFs
    vmin, vmax = -np.max(abs(eofs_masked)), np.max(abs(eofs_masked))
    
    for col in range(n_eofs):
        ax = axes[col]
        # Flip the EOF arrays vertically for proper orientation
        flipped_eof = np.flipud(eofs_masked[col])
        # Plot each EOF as an image
        im = ax.imshow(flipped_eof, cmap="bwr", interpolation='none', vmin=vmin, vmax=vmax)
        ax.set_title(f'EOF {col + 1}', fontsize=14)
        ax.axis('off')
    
    # Add a color bar below the subplots with a label
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.025, pad=0.04)
    cbar.set_label('Loading Value', fontsize=12)
    
    plt.suptitle(title, fontsize=16)
    plt.show()


def validate_pcs_plotter(pcs, title, size=(15, 5)):
    """
    Plot time series of the principal components (PCs).

    Parameters:
    - pcs (numpy.ndarray): A 2D numpy array of principal components with dimensions (time, n_pcs).
    - title (str): The title for the plot.
    - size (tuple, optional): A tuple specifying the figure size. Default is (15, 5).

    Returns:
    - None: The function generates a plot showing the time series of the principal components.
    """
    plt.figure(figsize=size)
    for i in range(pcs.shape[1]):
        plt.plot(pcs[:, i], label=f'PC {i+1}')
    plt.title(f'Time Series of {title}')
    plt.xlabel('Time')
    plt.ylabel('Principal Component Value')
    plt.legend()
    plt.show()


def validate_pcs(anomalies_normal, eofs_reshaped, pcs, lat, lon, year, period_train, season, n_eofs=7):
    """
    Validate the principal components (PCs) by comparing calculated PCs with reference time series.

    Parameters:
    - anomalies_normal (numpy.ndarray): A 3D numpy array of normalized precipitation anomalies with dimensions (time, lat, lon).
    - eofs_reshaped (numpy.ndarray): A 3D numpy array of reshaped EOFs with dimensions (n_eofs, lat, lon).
    - pcs (numpy.ndarray): A 2D numpy array of principal components with dimensions (time, n_eofs).
    - lat (numpy.ndarray): A 1D numpy array of latitudes.
    - lon (numpy.ndarray): A 1D numpy array of longitudes.
    - year (numpy.ndarray): A 1D numpy array of years corresponding to the time dimension of `anomalies_normal`.
    - period_train (tuple): A tuple specifying the training period as (start_year, end_year).
    - season (str): The season for which the analysis is performed.
    - n_eofs (int, optional): The number of EOFs to compute and validate. Default is 7.

    Returns:
    - None: The function generates and displays several plots comparing the calculated PCs with reference time series.
    """
    # Manual calculation of time series
    # NaN handling
    anomalies_normal_reshaped = anomalies_normal.reshape(len(year), len(lat) * len(lon))
    # Fill NaNs with 0
    anomalies_normal_reshaped_filled = np.nan_to_num(anomalies_normal_reshaped, nan=0.0)
    # Fill NaNs in eofs_reshaped with 0
    eofs_filled = np.nan_to_num(eofs_reshaped.reshape((n_eofs, len(lat) * len(lon))), nan=0.0)

    pcs_manual = np.dot(anomalies_normal_reshaped_filled, eofs_filled.T)

    # Get and plot reference time series
    filepath_fls = f'/nr/samba/PostClimDataNoBackup/CONFER/EASP/fls/chirps/seasonal/halfdeg_res/refper_1993-2020/prec_full_{season}.csv'
    data_prec_full = pd.read_csv(filepath_fls)

    # Select the rows where loy equals 1981
    selected_loy = 1981
    data_prec = data_prec_full[data_prec_full['loy'] == selected_loy]

    # Filter data to only include the first n_eofs EOFs
    data_prec = data_prec[data_prec['eof'] <= n_eofs]

    # Specify the years and EOFs to extract 
    train_period = np.arange(period_train[0], period_train[1] + 1) 
    selected_years = train_period

    # Extract 'fl' and 'sfl' values and reshape the data to (year, n_eofs)
    fl_values = data_prec.pivot(index='year', columns='eof', values='fl').reindex(selected_years, fill_value=np.nan).values
    sfl_values = data_prec.pivot(index='year', columns='eof', values='sfl').reindex(selected_years, fill_value=np.nan).values

    # Plotting normalized anomalies time series
    validate_pcs_plotter(pcs, 'Normalized Anomalies PCs', size=(10, 3))
    # Plotting manually calculated normalized anomalies time series
    validate_pcs_plotter(pcs_manual, 'Normalized Anomalies PCs - Manual', size=(10, 3))
    # Plotting reference normalized anomalies time series
    validate_pcs_plotter(fl_values, 'Normalized Anomalies PCs - Reference', size=(10, 3))
    # Plotting reference scaled normalized anomalies time series
    validate_pcs_plotter(sfl_values, 'Scaled Normalized Anomalies PCs - Reference', size=(10, 3))


def plot_time_series(df, index_name, comparison=False):
    """
    Plot the time series of the standardized anomaly index.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the time series data with columns 'year', 'month', and 'standardized_anomaly'.
                             If comparison is True, the DataFrame should also contain a 'fl' column for the reference values.
    - index_name (str): The name of the index being plotted (e.g., 'n34', 'dmi').
    - comparison (bool, optional): Whether to include a comparison with reference values. Default is False.

    Returns:
    - None: The function generates and displays a plot of the time series.
    """
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df = df.sort_values('date')

    plt.figure(figsize=(12, 6))
    if comparison:
        plt.plot(df['date'], df['standardized_anomaly'], label=f"{index_name} Calculated", color='b')
        plt.plot(df['date'], df['fl'], label=f"{index_name} Reference", color='r', linestyle='--')
    else:
        plt.plot(df['date'], df['standardized_anomaly'], label=f"{index_name} Index", color='b')
    plt.title(f"Time Series of {index_name} Index")
    plt.xlabel("Time")
    plt.ylabel(f"{index_name} Index Value")
    plt.legend()
    plt.grid(True)
    plt.show()
            

def validate_indices(era5_indices, filepath_indices, period_train, year_fcst):
    """
    Validates calculated indices against reference indices and prints comparisons for specified forecast months.

    Args:
    era5_indices (pd.DataFrame): Dataframe containing ERA5 indices with columns 'year', 'month', and index names.
    filepath_indices (str): Filepath prefix where reference index files are located.
    period_train (tuple): A tuple containing the start and end years (inclusive) for the training period.
    year_fcst (int): The forecast year for which the validation is being performed.

    Returns:
    None
    """
    for index in era5_indices.columns.difference(['year', 'month']):  # Skip 'year' and 'month' columns
        print(f"Validating index: {index}")

        # Extract the relevant columns for this index
        index_df = era5_indices[['year', 'month', index]].rename(columns={index: 'standardized_anomaly'})
        
        # Process the reference dataframe for the given index
        reference_df = pd.read_csv(f"{filepath_indices}{index}_full.csv")
        # Filter the reference dataframe for the specified time range
        reference_df_filtered = reference_df[(reference_df['year'] >= period_train[0]) & (reference_df['year'] <= period_train[1])][['year', 'month', 'fl']]
        # Drop duplicate rows
        reference_df_filtered = reference_df_filtered.drop_duplicates()
        
        # Merge and plot dataframes
        merged_df = pd.merge(index_df, reference_df_filtered, on=['year', 'month'], suffixes=('_calculated', '_reference'))
        plot_time_series(merged_df, index, comparison=True)

        # Print some values for comparison
        for month in range(1, 5):
            print(f"Standardized anomaly (calculated index) for forecast year: {year_fcst}, forecast month = {month} (based on data from month before):")
            
            # Calculate the value for the previous month
            prev_year, prev_month = (year_fcst - 1, 12) if month == 1 else (year_fcst, month - 1)
            # Get the calculated and reference values
            calculated_value = era5_indices[(era5_indices['year'] == prev_year) & (era5_indices['month'] == prev_month)][index].values[0]
            reference_value = reference_df[(reference_df["year"] == prev_year) & (reference_df["month"] == prev_month)].fl.values[0]
            print(f'Calculated value: {calculated_value}')
            print(f'Reference value: {reference_value}')
            print("\n")


def validate_ml_coefficients(df_coefficients, period_train, season, month_init):
    """
    Validate and visualize the machine learning model coefficients.

    This function generates a heatmap to visualize the coefficients of a machine learning model
    for a specified training period, season, and initialization month.

    Parameters:
    - df_coefficients (pd.DataFrame): DataFrame containing the model coefficients.
                                      The rows represent EOFs, and the columns represent features.
    - period_train (tuple): Tuple containing the start and end years of the training period (e.g., (1981, 2020)).
    - season (str): The season for which the model coefficients are being visualized. 
                    Expected values: 'MAM', 'JJAS', 'OND'.
    - month_init (int): The initialization month for the model predictions. Expected values: 1 to 12.

    Returns:
    - None: The function displays a heatmap of the model coefficients.
    """
    # Define month string dictionary
    month_str = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }[month_init]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the coefficients
    img = ax.imshow(df_coefficients, vmin={'MAM': -80., 'JJAS': -160., 'OND': -230.}[season],
                    vmax={'MAM': 80., 'JJAS': 160., 'OND': 230.}[season], cmap='bwr',
                    extent=[0, len(df_coefficients.columns), 0, len(df_coefficients.index)])

    ax.set_xticks(np.arange(0.5, len(df_coefficients.columns)))
    ax.set_xticklabels(df_coefficients.columns.to_list(), rotation=90, fontsize=10)
    ax.set_yticks(np.arange(len(df_coefficients.index) - 0.5, 0, -1))
    ax.set_yticklabels(df_coefficients.index.to_list(), fontsize=10)
    ax.set_title(f'CHIRPS, training period: {period_train[0]}-{period_train[1]}', fontsize=16)

    # Adjust layout and add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    fig.colorbar(img, cax=cbar_ax)

    # Display the plot
    plt.show()


def validate_forecast(prob_bn, prob_an, prec_data, period_clm, period_train, year, lat, lon, season, year_fcst):
    """
    Validate the precipitation forecast against actual data by comparing predicted categories with actual categories.

    Parameters:
    - prob_bn (numpy.ndarray): Array of probabilities for below-normal precipitation.
    - prob_an (numpy.ndarray): Array of probabilities for above-normal precipitation.
    - prec_data (numpy.ndarray): 3D array of precipitation data with dimensions (time, lat, lon).
    - ref_period_indices (list): List of indices corresponding to the reference period for calculating percentiles.
    - period_train (tuple): Tuple containing the start and end years of the training period (e.g., (1981, 2020)).
    - period_clm (tuple): Tuple containing the start and end years of the reference period (e.g., (1993, 2020)).
    - year (numpy.ndarray): A 1D numpy array of years corresponding to the time dimension of `anomalies_normal`.
    - lat (numpy.ndarray): 1D array of latitude values.
    - lon (numpy.ndarray): 1D array of longitude values.
    - season (str): The season for which the forecast is being validated (e.g., 'MAM', 'JJAS', 'OND').
    - year_fcst (int): The year for which the forecast is being validated.

    Returns:
    - None: This function displays a plot showing the validation of the forecast.
    """
    # Get reference period indices
    ref_period_mask = (year >= period_clm[0]) & (year <= period_clm[1])
    ref_period_indices = np.where(ref_period_mask)[0]

    # Calculate the 33rd and 67th percentiles for each grid point for the reference period
    percentile_33 = np.nanpercentile(prec_data[ref_period_indices, :, :], 33, axis=0)
    percentile_67 = np.nanpercentile(prec_data[ref_period_indices, :, :], 67, axis=0)

    # Select the specific year
    actual_precip = prec_data[year_fcst - period_train[0], :, :]

    # Categorize the precipitation and create a categorical array: 0 for below normal, 1 for normal, 2 for above normal
    actual_categories = xr.where(actual_precip < percentile_33, 0, xr.where(actual_precip > percentile_67, 2, 1))

    # Determine the predicted category based on highest probability and compare with actual categories
    verification = xr.where(prob_bn > 0.4, 0, xr.where(prob_an > 0.4, 2, 1)) == actual_categories

    # Reintroduce NaNs based on the original prec_data
    masked_verification = np.ma.masked_where(np.isnan(prec_data[year_fcst - period_train[0], :, :]), verification)

    # Create a custom colormap
    cmap = mcolors.ListedColormap(['red', 'green', 'gray'])
    bounds = [0, 0.5, 1.5, 2]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={'projection': None})

    im = ax.imshow(masked_verification, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                   origin='lower', cmap=cmap, norm=norm)

    ax.set_title(f'Verification of {season} {year_fcst} Precipitation Forecast')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Create a color bar with the correct labels
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', ticks=[0.25, 1, 1.75])
    cbar.ax.set_yticklabels(['Incorrect', 'Correct', 'Masked'])

    plt.tight_layout()
    plt.show()


