
import numpy as np
import matplotlib.pyplot as plt

#import pickle

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


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









