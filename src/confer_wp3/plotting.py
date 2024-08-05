import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_fields_simple(fields, titles, cmap, unit, lat, lon, season, year):
    """
    Plot multiple fields on a single figure with individual colorbars.

    This function generates a side-by-side comparison of different fields (e.g., predicted probabilities for different terciles)
    on a map, each with its own colorbar. The fields are displayed using the specified colormaps and titles.

    Parameters:
    - fields (list of numpy.ndarray): List of 2D arrays to be plotted. Each array represents a field (e.g., predicted probabilities).
    - titles (list of str): List of titles for each subplot.
    - cmap (list of str or Colormap): List of colormaps to be used for each field.
    - unit (str): Unit label for the colorbars.
    - season (str): Season for which the fields are being plotted (e.g., 'MAM', 'JJAS', 'OND').
    - year (int): Year for which the fields are being plotted.

    Returns:
    - None: This function displays a plot with multiple subplots, each showing a different field with a colorbar.
    """
    n_fields = len(fields)
    fig, axes = plt.subplots(1, n_fields, figsize=(15, 5), subplot_kw={'projection': None})

    for i, ax in enumerate(axes):
        im = ax.imshow(fields[i], extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                       origin='lower', cmap=cmap[i], vmin=0.333, vmax=1)
        ax.set_title(titles[i])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        cbar = fig.colorbar(im, ax=ax, orientation='vertical')
        cbar.set_label(unit)

    fig.suptitle(f'Predicted tercile probabilities for {season} precipitation amounts, {year}', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_combined_terciles(prob_bn, prob_an, lat, lon, season, year):
    """
    Plot the combined tercile probabilities on a single map with three color scales.

    Parameters:
    - prob_bn (numpy.ndarray): 2D array of probabilities for below-normal precipitation.
    - prob_an (numpy.ndarray): 2D array of probabilities for above-normal precipitation.
    - lat (numpy.ndarray): 1D array of latitudes.
    - lon (numpy.ndarray): 1D array of longitudes.
    - season (str): Season for which the fields are being plotted.
    - year (int): Year for which the fields are being plotted.
    """

    # Calculate normal probability
    prob_nn = 1 - (prob_bn + prob_an)
    
    # Create custom colormaps for each tercile
    cmap_bn = LinearSegmentedColormap.from_list('custom_bn', ['#FFFFFF', '#FF0000'], N=100)
    cmap_nn = LinearSegmentedColormap.from_list('custom_nn', ['#FFFFFF', '#FFFF00'], N=100)
    cmap_an = LinearSegmentedColormap.from_list('custom_an', ['#FFFFFF', '#008000'], N=100)
    
    # Create a masked array for each tercile
    mask_bn = np.ma.masked_where(prob_bn <= np.maximum(prob_nn, prob_an), prob_bn)
    mask_nn = np.ma.masked_where(prob_nn <= np.maximum(prob_bn, prob_an), prob_nn)
    mask_an = np.ma.masked_where(prob_an <= np.maximum(prob_bn, prob_nn), prob_an)

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each tercile
    im_bn = ax.imshow(mask_bn, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                      origin='lower', cmap=cmap_bn, vmin=0.33, vmax=1)
    im_nn = ax.imshow(mask_nn, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                      origin='lower', cmap=cmap_nn, vmin=0.33, vmax=1)
    im_an = ax.imshow(mask_an, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                      origin='lower', cmap=cmap_an, vmin=0.33, vmax=1)
    
    ax.set_title(f'Tercile Probabilities for {season} Precipitation, {year}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Create custom colorbars
    cax_bn = fig.add_axes([0.92, 0.7, 0.02, 0.2])
    cbar_bn = fig.colorbar(im_bn, cax=cax_bn, orientation='vertical')
    cbar_bn.set_label('Below Normal')
    
    cax_nn = fig.add_axes([0.92, 0.4, 0.02, 0.2])
    cbar_nn = fig.colorbar(im_nn, cax=cax_nn, orientation='vertical')
    cbar_nn.set_label('Normal')
    
    cax_an = fig.add_axes([0.92, 0.1, 0.02, 0.2])
    cbar_an = fig.colorbar(im_an, cax=cax_an, orientation='vertical')
    cbar_an.set_label('Above Normal')
    
    # Set colorbar ticks
    for cbar in [cbar_bn, cbar_nn, cbar_an]:
        cbar.set_ticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    # Add legend
    legend_elements = [Rectangle((0, 0), 1, 1, fc='#FF0000', label='Drier'),
                       Rectangle((0, 0), 1, 1, fc='#FFFF00', label='Average'),
                       Rectangle((0, 0), 1, 1, fc='#008000', label='Wetter')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.show()



def get_nearest_grid_index(lon_exmpl, lat_exmpl, lon_grid, lat_grid):
    ix = np.argmin(abs(lon_grid-lon_exmpl))
    iy = np.argmin(abs(lat_grid-lat_exmpl))
    return ix, iy

def get_xticks(x_extent, inc = 1):
    x_inc = np.arange(-180,180,inc)
    return(x_inc[np.where(np.logical_and(x_inc >= x_extent[0], x_inc <= x_extent[1]))])

def get_yticks(y_extent, inc = 1):
    y_inc = np.arange(-90,90,inc)
    return(y_inc[np.where(np.logical_and(y_inc >= y_extent[0], y_inc <= y_extent[1]))])



def plot_fields (fields_list, lon, lat, lon_bounds, lat_bounds, main_title, subtitle_list, unit, vmin=None, vmax=None, cmap='BuPu', water_bodies=False, ticks=True, tick_labels=None):

    n_img = len(fields_list)
    img_extent = lon_bounds + lat_bounds

    if not type(unit) is list:
        unit = [unit for i in range(n_img)]

    if not type(cmap) is list:
        cmap = [cmap for i in range(n_img)]

    if vmin == None:
        vmin = [np.nanmin(field) for field in fields_list]
    if vmax == None:
        vmax = [np.nanmax(field) for field in fields_list]

    if not type(vmin) is list:
        vmin = [vmin for i in range(n_img)]
    if not type(vmax) is list:
        vmax = [vmax for i in range(n_img)]       

    if ticks == True:
        ticks = [ticks for i in range(n_img)]
    elif not type(ticks) is list:
        print("Error! Argument 'ticks' must be a list or a list of lists.")
    elif all([isinstance(tt, float) or isinstance(tt, int) for tt in ticks]):
        ticks = [ticks for i in range(n_img)]

    if tick_labels == None:
        tick_labels = [tick_labels for i in range(n_img)]
    elif not type(tick_labels) is list:
        print("Error! Argument 'ticks_labels' must be a list or a list of lists.")
    elif all([isinstance(tt, float) or isinstance(tt, int) or isinstance(tt, str) for tt in tick_labels]):
        tick_labels = [tick_labels for i in range(n_img)]

    r = abs(lon[1]-lon[0])
    lons_mat, lats_mat = np.meshgrid(lon, lat)
    lons_matplot = np.hstack((lons_mat - r/2, lons_mat[:,[-1]] + r/2))
    lons_matplot = np.vstack((lons_matplot, lons_matplot[[-1],:]))
    lats_matplot = np.hstack((lats_mat, lats_mat[:,[-1]]))
    lats_matplot = np.vstack((lats_matplot - r/2, lats_matplot[[-1],:] + r/2))     # assumes latitudes in ascending order

    dlon = (lon_bounds[1]-lon_bounds[0]) // 8
    dlat = (lat_bounds[1]-lat_bounds[0]) // 8

    fig_height = 7.
    fig_width = (n_img*1.15)*(fig_height/1.1)*np.diff(lon_bounds)[0]/np.diff(lat_bounds)[0]

    fig = plt.figure(figsize=(fig_width,fig_height))
    for i_img in range(n_img):
        ax = fig.add_subplot(100+n_img*10+i_img+1, projection=ccrs.PlateCarree())
        cmesh = ax.pcolormesh(lons_matplot, lats_matplot, fields_list[i_img], cmap=cmap[i_img], vmin=vmin[i_img], vmax=vmax[i_img])
        ax.set_extent(img_extent, crs=ccrs.PlateCarree())
        ax.set_yticks(get_yticks(img_extent[2:4],dlat), crs=ccrs.PlateCarree())
        ax.yaxis.set_major_formatter(LatitudeFormatter()) 
        ax.set_xticks(get_xticks(img_extent[0:2],dlon), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        ax.add_feature(cfeature.COASTLINE, linewidth=2)
        ax.add_feature(cfeature.BORDERS, linewidth=2, linestyle='-', alpha=.9)
        if water_bodies:
            ax.add_feature(cfeature.LAKES, alpha=0.95)
            ax.add_feature(cfeature.RIVERS)

        plt.title(subtitle_list[i_img], fontsize=14)
        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
        fig.add_axes(ax_cb)
        cbar = plt.colorbar(cmesh, cax=ax_cb)
        cbar.set_label(unit[i_img])
        if not ticks[i_img] == True:
            cbar.set_ticks(ticks[i_img])
            cbar.set_ticklabels(tick_labels[i_img])

    fig.canvas.draw()
    plt.tight_layout(rect=[0,0,1,0.95])
    fig.suptitle(main_title, fontsize=16)
    plt.show()









