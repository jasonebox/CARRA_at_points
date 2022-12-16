#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 06:43:31 2022

@author: jason
"""

import numpy as np
import os
from netCDF4 import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import calendar

ch=2 # tp
# ch=1 # rf
# ch=0 # t2m
prt_time=0
tst_plot=0

path='/Users/jason/0_dat/CARRA_at_points/'
# raw_path='/Users/jason/0_dat/CARRA_raw/'
# # raw_path='/Volumes/LaCie/0_dat/CARRA/CARRA_raw/'
# raw_path='/Volumes/Samsung_T5/0_dat/CARRA/CARRA_raw/'
# outpath='/Users/jason/0_dat/CARRA_at_TIN_points/3h/' 
# # outpath='/Volumes/LaCie/0_dat/CARRA/output/annual/' 

os.chdir(path)

years=np.arange(1991,2020).astype('str')

fn='./site_coords/TIN_Greenland_ccordinates_for_CARRA.csv'
df=pd.read_csv(fn)
n=len(df)

for yy,year in enumerate(years):
    if year=='1991':
        if calendar.isleap(int(year)):
            print(year+' is leap year')
            n_days=366
        else:
            print(year+' is not a leap year')
            n_days=365
            
        with open('./3h/'+year+'.npy', 'rb') as f:
            rf_3h_5dcube=np.load(f)
            tp_3h_5dcube=np.load(f)
            t2m_3h_5dcube=np.load(f)
        
        location=1 # sisimiut_1
        location=2 # sisimiut_2
        print(np.shape(rf_3h_5dcube))
        sumx=np.zeros((np.shape(t2m_3h_5dcube)[3],np.shape(rf_3h_5dcube)[4]))
        # for dd in range(np.shape(rf_3h_5dcube)[2]):
        for dd in range(n_days):
            for hh in range(8):
                # sumx=t2m_3h_5dcube[location,hh,dd,:,:]
        
                sumx+=tp_3h_5dcube[location,hh,dd,:,:] # totaling precip
        plt.close()
        plt.imshow(sumx)
        plt.colorbar()
        plt.title(df.name[location]+' tp '+year)
        # plt.title(df.name[location]+' tp\nday of year '+str(dd+1)+' '+str((hh+1)*3)+'h UTC')
        # plt.axis('off')
        ly='x'
        if ly == 'x':
            plt.show()
