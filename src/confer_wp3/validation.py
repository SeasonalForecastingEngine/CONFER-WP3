"""
This file contains code for validating the calculations made in lasso_fcst_example. 
The code is not needed to run the forecast.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import norm

def validate_anomalies1(prec_data, anomalies, lat, lon, year_index = 0):
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
    print("Visualize the original precipitation data and calculated anomalies on a map for a single year.")
    # Plot original precipitation data
    plt.figure(figsize=(8, 10))  # Adjusting figsize for a more even aspect ratio
    plt.subplot(2, 1, 1)
    plt.title('Original Precipitation Data')
    plt.imshow(prec_data[year_index, :, :], origin='lower', extent=[lon.min(), lon.max(), lat.min(), lat.max()], cmap='viridis', aspect='auto')
    plt.colorbar(label='Precipitation (mm)')

    # Plot anomalies
    plt.subplot(2, 1, 2)
    plt.title('Precipitation Anomalies')
    plt.imshow(anomalies[year_index, :, :], origin='lower', extent=[lon.min(), lon.max(), lat.min(), lat.max()], cmap='bwr', aspect='auto')
    plt.colorbar(label='Anomalies (mm)')
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
    print("Visualize and compare original and transformed precipitation anomalies for a single year.")
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
    print("Visualize and compare the distribution of original and transformed precipitation anomalies using histograms.")
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

