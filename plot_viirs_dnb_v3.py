import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import h5py
from pyhdf.SD import SD, SDC
import glob
import pandas as pd
from netCDF4 import Dataset 
import os
import datetime
from math import *
import mysatellite as ms
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt
from more_itertools import locate
from matplotlib.colors import LinearSegmentedColormap
import joblib
from collections import deque
from array import *
from IPython.display import HTML
from scipy.io import *
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
from pylab import figure, cm

from cartopy import config
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
from pyresample import kd_tree
from pyresample import utils
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PIL import Image 

def save_viirs(viirs_file='',timeflag='',sav_path=''):

    nc_fid = Dataset(viirs_file, 'r')
    rad = nc_fid['/observation_data/DNB_observations'][:]
    nc_fid.close()

    n_w = 1000
    n_h = 1000

    for i in range(3):
        for j in range(4):
            ii = i*n_h + 160
            jj = j*n_w + 32
            sub_rad = rad[ii:ii+n_h,jj:jj+n_w]
            sav_file = sav_path + timeflag + '_' + str(j).zfill(1) + str(i).zfill(1) + '.h5'
            #sub_rad[sub_rad<0] = 1.0e-15
            #vmin = max(np.percentile(sub_rad, 5),1.0e-15)
            #vmax = min(np.percentile(sub_rad, 95),1.0e-3)

            #sub_rad_log = np.log(sub_rad)
            #vmin_log = log(vmin)
            #vmax_log = log(vmax)
            #sub_rad_plot = np.zeros([1000,1000],dtype=np.uint8)
            #sub_rad_plot = np.uint8((sub_rad_log-vmin_log)*255.0/(vmax_log-vmin_log))

            try:
                #plt.figure(figsize=(n_w/my_dpi, n_h/my_dpi), dpi=my_dpi)
                #plt.matshow(sub_rad, cmap=cmap, norm=LogNorm(vmin=min(vmin,vmax), vmax=max(vmin,vmax)))
                #fig ,ax = plt.subplots(figsize = (18,18))
                #plt.imshow(sub_rad_log,cmap='gray')
                #plt.matshow(sub_rad, cmap=cmap, norm=LogNorm(vmin=min(vmin,vmax), vmax=max(vmin,vmax)))
                #plt.gca().set_xticks([])
                #plt.gca().set_yticks([])
                #plt.savefig(sav_file,bbox_inches='tight',pad_inches=-0.1)
                sid = h5py.File(sav_file,'w')
                sid.create_dataset('DNB_observations',data=sub_rad)
                sid.close() 
            except:
                pass


def plot_viirs(viirs_file='',timeflag='',sav_path=''):
    
    nc_fid = Dataset(viirs_file, 'r')
    rad = nc_fid['/observation_data/DNB_observations'][:]
    nc_fid.close()
    
    cmap = cm.gray
    cmap.set_bad(color = 'black', alpha = 1.)
    cmap.set_under(color = 'black', alpha = 1.)
    cmap.set_over(color = 'white', alpha = 1.)

    n_w = 1000
    n_h = 1000
    #my_dpi = 96

    for i in range(3):
        for j in range(4):
            ii = i*n_h + 160
            jj = j*n_w + 32
            sub_rad = rad[ii:ii+n_h,jj:jj+n_w]
            sav_file = sav_path + timeflag + '_' + str(j).zfill(1) + str(i).zfill(1) + '.png'
            sub_rad[sub_rad<0] = 1.0e-15
            vmin = max(np.percentile(sub_rad, 5),1.0e-15)
            vmax = min(np.percentile(sub_rad, 95),1.0e-3)

            sub_rad_log = np.log(sub_rad)
            #vmin_log = log(vmin)
            #vmax_log = log(vmax)
            #sub_rad_plot = np.zeros([1000,1000],dtype=np.uint8)
            #sub_rad_plot = np.uint8((sub_rad_log-vmin_log)*255.0/(vmax_log-vmin_log))

            try:
                #plt.figure(figsize=(n_w/my_dpi, n_h/my_dpi), dpi=my_dpi)
                #plt.matshow(sub_rad, cmap=cmap, norm=LogNorm(vmin=min(vmin,vmax), vmax=max(vmin,vmax)))
                fig ,ax = plt.subplots(figsize = (18,18))
                #plt.imshow(sub_rad_log,cmap='gray')
                plt.matshow(sub_rad, cmap=cmap, norm=LogNorm(vmin=min(vmin,vmax), vmax=max(vmin,vmax)))
                plt.gca().set_xticks([])
                plt.gca().set_yticks([])
                plt.savefig(sav_file,bbox_inches='tight',pad_inches=-0.1)
            except:
                pass
    return

list_file = './vnb_list.txt'
with open(list_file) as f:
    viirs_files = f.read().splitlines()

for viirs_file in viirs_files:
    print (viirs_file)
    viirs_name = os.path.basename(viirs_file)
    viirs_timeflag = viirs_name[10:22]
    #plot_viirs(viirs_file=viirs_file,timeflag=viirs_timeflag,sav_path='./')
    save_viirs(viirs_file=viirs_file,timeflag=viirs_timeflag,sav_path='./vnb_patch_data_2017/')
#viirs_file = '/tis/modaps/allData/5110/VNP02DNB/2017/001/VNP02DNB.A2017001.0624.001.2017277094015.nc'
#viirs_file = '/tis/modaps/allData/5110/VNP02DNB/2017/120/VNP02DNB.A2017120.1918.001.2017278233334.nc'
#viirs_name = os.path.basename(viirs_file)
#viirs_timeflag = viirs_name[10:22]
#plot_viirs(viirs_file=viirs_file,timeflag=viirs_timeflag,sav_path='./vnb_patch/')