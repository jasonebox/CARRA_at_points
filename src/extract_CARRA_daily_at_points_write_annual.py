#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors:
    Jason Box and Adrien Wehrlé

updated Dec 2022
Jason Box, GEUS, jeb@geus.dk
"""

# import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import xarray as xr
# import glob
import geopandas as gpd
from scipy.spatial import distance
from shapely.geometry import Point
from datetime import datetime, timedelta
import calendar

# global plot settings
font_size=20
th=1
# plt.rcParams['font.sans-serif'] = ['Georgia']
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 1
co=0.9; plt.rcParams['grid.color'] = (co,co,co)
plt.rcParams["font.size"] = font_size
plt.rcParams['legend.fontsize'] = font_size*0.8
plt.rcParams['mathtext.default'] = 'regular'


if os.getlogin() == 'jason':
    base_path = '/Users/jason/Dropbox/CARRA/CARRA_at_points/'
    # base_path = '/Volumes/LaCie/CARRA/CARRA_rain/'
os.chdir(base_path)
iyear=1991 ; fyear=2021

#%%
# varname=['t2m'] ; app='PROMICE'
# varname=['t2m'] ; app='TAS'
# varname=['t2m'] ; app='t-10m'
# # varname=['rf'] ; app='PROMICE'
# varname=['t2m','tp'] ; app='points'
# # varname=['t2m'] ; app='Zackenberg'
# # varname=['rf'] ; app='Zackenberg'
# varname=['tp','rf','t2m'] ; app='Zackenberg'
# # varname=['tp','rf','t2m'] ; app='GNET'
# varname=['tp','rf','t2m'] ; app='GEUS_AWS'
# # varname=['t2m'] ; app='GEUS_AWS'
# # app='RCM_validation'
# # varname=['tp','rf','t2m','bss','swgf'] ; app='points'

# varname=['tp','rf','t2m'] ; app=='GC-Net'
varname=['tp','rf','t2m'] ; app='GC-Net'

# for varnam in varname:

for varnam in varname:
# for varnam in varname[2:3]:
# for varnam in varname[3:4]:
# for varnam in varname[4:5]: #SWGF
       
    

 
    
    def lon360_to_lon180(lon360):
    
        # reduce the angle  
        lon180 =  lon360 % 360 
        
        # force it to be the positive remainder, so that 0 <= angle < 360  
        lon180 = (lon180 + 360) % 360;  
        
        # force into the minimum absolute value residue class, so that -180 < angle <= 180  
        lon180[lon180 > 180] -= 360
        
        return lon180
    
    # CARRA West grid dims
    ni = 1269 ; nj = 1069
    
    # read lat lon arrays
    fn = './ancil/2.5km_CARRA_west_lat_1269x1069.npy'
    lat = np.fromfile(fn, dtype=np.float32)
    lat_mat = lat.reshape(ni, nj)
    lat_mat=np.rot90(lat_mat.T)
    
    fn = './ancil/2.5km_CARRA_west_lon_1269x1069.npy'
    lon = np.fromfile(fn, dtype=np.float32)
    lon_pn = lon360_to_lon180(lon)
    lon_mat = lon_pn.reshape(ni, nj) 
    lon_mat=np.rot90(lon_mat.T)
    
    # if app=='snow_pits':
    #     meta = pd.read_excel('./ancil/Greenland_snow_pit_SWE_vfyear0409.xlsx')
    #     # having a column named "name" create conflicts in DataFrame
    #     meta.rename(columns={'Name': 'station_name'}, inplace=True)
    #     meta = meta.drop(meta[meta["End Year"] <1998].index)
    #     # print(meta.columns)
    #     # print(len(meta))
    
    if app=='PROMICE':
        # read PROMICE locations
        meta = pd.read_csv('./ancil/PROMICE_info_w_header_2017-2018_stats_w_Narsaq_etc.csv',delim_whitespace=True)
        # meta = pd.read_csv('./ancil/PROMICE_TAS.csv',delim_whitespace=True)
        # having a column named "name" create conflicts in DataFrame
        meta.rename(columns={'name': 'station_name'}, inplace=True)

    if app=='t-10m':
        # read PROMICE locations
        meta = pd.read_csv('/Users/jason/Dropbox/BAV firn T study/coords.csv')
        # meta = pd.read_csv('./ancil/PROMICE_TAS.csv',delim_whitespace=True)
        # having a column named "name" create conflicts in DataFrame
        meta.rename(columns={'site': 'station_name'}, inplace=True)
    
    if app=='Zackenberg':
        # read PROMICE locations
        meta = pd.read_csv('./ancil/'+app+'.csv')

    if app=='GEUS_AWS':
        # read PROMICE locations
        meta = pd.read_csv('/Users/jason/Dropbox/AWS/_merged/GEUS_AWS_sites_20221128.csv')   
        meta.lon=-meta.lon

    if app=='GC-Net':
        # read PROMICE locations
        meta = pd.read_csv('/Users/jason/Dropbox/AWS/GCNET/GCNet_positions/output/GC-Net_info_incl_1999.csv')   
        meta.rename(columns={'name': 'station_name'}, inplace=True)

    if app=='GNET':
        # read PROMICE locations
        meta = pd.read_csv('/Users/jason/Dropbox/GNET/GNET_stations.csv')   

    # if app=='RCM_validation':
    #     meta = pd.read_csv('./ancil/Greenland_sites_for_eccum_rain_evaluation_w_PROMICE_etc.csv')
    #     meta.rename(columns={'Name': 'station_name'}, inplace=True)

    # meta.replace('#REF!', np.nan, inplace=True)
    
    if app=='points':
        meta = pd.read_csv('./ancil/Greenland_sites_for_accum_rain_evaluation_w_PROMICE_etc_vfyear0929.csv')
        meta.rename(columns={'Name': 'station_name'}, inplace=True)

    
    # %% point extraction takes 25 sec
    print('point extraction takes 25 sec')
    def match(station_point, CARRA_points):
        
        dists = distance.cdist(station_point, CARRA_points)
        
        dist = np.nanmin(dists)
        
        idx = dists.argmin()
        
        station_gdfpoint = Point(station_point[0, 0], station_point[0, 1])
        matching_CARRA_cell = CARRA_gdfpoints.loc[idx]['geometry']
        
        res = gpd.GeoDataFrame({'promice_station': [station_gdfpoint], 
                                'CARRA_cell': [matching_CARRA_cell],
                                'distance': pd.Series(dist)})
        
        return res, idx
    
    cols, rows = np.meshgrid(np.arange(np.shape(lat_mat)[1]), 
                             np.arange(np.shape(lat_mat)[0]))
    
    CARRA_positions = pd.DataFrame({'row': rows.ravel(),
                                    'col': cols.ravel(),
                                    'lon': lon_mat.ravel(),
                                    'lat': lat_mat.ravel()})
    
    CARRA_points = np.vstack((CARRA_positions.lon.ravel(), 
                              CARRA_positions.lat.ravel())).T
    
    CARRA_gdfpoints = gpd.GeoDataFrame(geometry=gpd.points_from_xy(CARRA_positions.lon, 
                                                                   CARRA_positions.lat))
    
    
    # match point locations with CARRA cells 
    
    CARRA_rows = []
    CARRA_cols = []
    
    CARRA_cells_atPROMICE = {}
    
    for r, station in meta.iterrows(): 
        
        station_point = np.vstack((station.lon,
                                    station.lat)).T
    
        # get CARRA cell matching station location
        CARRA_matching_cell, idx = match(station_point, CARRA_points)
        CARRA_matching_rowcol = CARRA_positions.iloc[idx]
        
        CARRA_cells_atPROMICE[station.station_name] = CARRA_matching_rowcol
        
        
    # %% Check the colocation on a map
    
    # fn='H:/CARRA/t2m_2016.nc'
    # if AW==0:fn='/Users/jason/0_dat/CARRA/output/t2m_2012.nc'
    # xds = xr.open_dataset(fn)
    
    # for station in CARRA_cells_atPROMICE.keys():
        
    #     r = int(CARRA_cells_atPROMICE[station].row)
    #     c = int(CARRA_cells_atPROMICE[station].col)
    
    #     xds["t2m"][0, r, c] = 50
        
    #     print(station)
                   
    # plt.figure()
    # plt.imshow(xds["t2m"][0, :, :], origin='lower')
        
    # %% time series at locations
    
    AW = 0
    
    if AW:
        CARRA_path = 'H:/CARRA/'
    if not AW:
        CARRA_path = '/Users/jason/0_dat/CARRA/output/'
        CARRA_path = '/Volumes/LaCie/CARRA/output/'
        CARRA_path = '/Users/jason/0_dat/CARRA/output/annual/'
    
    # ds = xr.open_dataset('H:/CARRA/tp_2016.nc')
        
    # years=np.arange(2019,fyear).astype('str')
    # years=np.arange(2001,fyear).astype('str')
    # years=np.arange(2000,2001).astype('str')
    # years=np.arange(1997,1998).astype('str')
    years=np.arange(iyear,fyear+1).astype('str')
    
    # results = pd.DataFrame()
    
    for i, year in enumerate(years):
        
        # print(varnam,i,year)
        
        # if ((varname!='swgf')&(varname!='bss')):
        #     ds = xr.open_dataset(CARRA_path+varname+'_'+str(year)+'.nc')                
        #     time = np.arange(datetime(int(year), 1, 1), datetime(int(year) + 1, 1, 1), 
        #                  timedelta(days=1)).astype(datetime)
        # else:
        #     ds = xr.open_dataset(CARRA_path+varname+'_2019.nc')
        #     time = np.arange(datetime(int(year), 1, 1), datetime(int(year) + 1, 1, 1), 
        #                  timedelta(days=1)).astype(datetime)
        # if varnam=='swgf':
        if ((varnam=='bss')or(varnam=='swgf')):
            ds = xr.open_dataset(CARRA_path+varnam+'_2019.nc',engine='h5netcdf')
            # print('hi')
        else:
            ds = xr.open_dataset(CARRA_path+varnam+'_'+year+'.nc')#,engine='h5netcdf')
            # print(ds)
            
        if bool(calendar.isleap(int(year))):
            time = np.arange(datetime(int(year), 1, 1), datetime(int(year) + 1, 1, 1), 
                      timedelta(days=1)).astype(datetime)
            # print(year,'leap',len(time))
        else:
            time = np.arange(datetime(int(year), 1, 1), datetime(int(year) + 1, 1, 1), 
                          timedelta(days=1)).astype(datetime)
            # print(year,'not leap',len(time))

        # time = np.arange(datetime(int(year), 1, 1), datetime(int(year) + 1, 1, 1), 
        #                   timedelta(days=1)).astype(datetime)    
        annual_results = pd.DataFrame()
            
        dt_time = pd.to_datetime(time)
        # print(len(dt_time))
        # if (((varname!='swgf')or(varname!='bss'))&(bool(calendar.isleap(int(year))))):
        #     dt_time=dt_time[0:365]

        # if bool(calendar.isleap(int(year))):
        #     dt_time=dt_time[0:365]

        annual_results['time'] = dt_time
        # print(i,year,len(dt_time))
        for r, station in meta.iterrows():
            
            print(varnam,year,len(meta)-r)
            
            CARRA_location = CARRA_cells_atPROMICE[station.station_name]
            
            # target time series at the point of interest
            if varnam=='tp':
                CARRA_PROMICE_timeseries = np.array(ds.tp[:,int(CARRA_location.row), 
                                                      int(CARRA_location.col)])
            # target time series at the point of interest
            if varnam=='rf':
                CARRA_PROMICE_timeseries = np.array(ds.rf[:,int(CARRA_location.row), 
                                                      int(CARRA_location.col)])        
            # target time series at the point of interest
            if varnam=='t2m':
                # temp=np.array(ds.t2m
                CARRA_PROMICE_timeseries = np.array(ds.t2m[:,int(CARRA_location.row), 
                                                      int(CARRA_location.col)])        

            # # target time series at the point of interest
            # if varnam=='bss':
            #     CARRA_PROMICE_timeseries = np.array(ds.bss[:,int(CARRA_location.row), 
            #                                           int(CARRA_location.col)])    

            # # target time series at the point of interest
            # if varnam=='swgf':
            #     CARRA_PROMICE_timeseries = np.array(ds.swgf[:,int(CARRA_location.row), 
            #                                           int(CARRA_location.col)])  
            
            # print(len(ds.t2m))
            # if (((varnam!='bss'))&(bool(calendar.isleap(int(year))))):
            #     CARRA_PROMICE_timeseries=np.append(CARRA_PROMICE_timeseries,np.array(0))
# len(CARRA_PROMICE_timeseries)

            nwtf=len(CARRA_PROMICE_timeseries)
            # print(nwtf)
            oswtf=0
            # if bool(calendar.isleap(int(year))):oswtf=1
            annual_results[station.station_name] = CARRA_PROMICE_timeseries[0:nwtf-oswtf]
            #         if bool(calendar.isleap(int(year))):
            #     CARRA_PROMICE_timeseries=np.array([CARRA_PROMICE_timeseries,0],order='F')
            #     CARRA_PROMICE_timeseries=np.resize(CARRA_PROMICE_timeseries,366)
            #     plt.plot(CARRA_PROMICE_timeseries)
            #     CARRA_PROMICE_timeseries.shape
            #     annual_results[station.station_name] = CARRA_PROMICE_timeseries[:],0]
            # else:

        # results = results.append(annual_results)
        opath='/Users/jason/Dropbox/CARRA/CARRA_at_points/output_csv/'
        # opath='/Users/jason/Dropbox/BAV firn T study/'
        
        annual_results.to_csv(opath+'CARRA_'+varnam+'_at_'+app+'_'+str(year)+'.csv',
                              index=None, float_format='%.2f')
        annual_results.to_csv('/Users/jason/0_dat/CARRA_temp/CARRA_'+varnam+'_at_points_'+str(year)+'.csv')

    #%%
cat_files=1

years=np.arange(iyear,fyear+1).astype('str')
varname=['tp','rf','t2m']
# varname=['t2m']
for varnam in varname:
    dfc = pd.DataFrame()
    if cat_files:
         for year in years:
             print(year)
             fn=opath+'CARRA_' + varnam+'_at_'+app+'_'+year+'.csv'
             # df=pd.read_csv(('./output_csv/' + varnam+'_at_'+app+'_'+year+'.csv'))
             df=pd.read_csv(fn)
             dfc = dfc.append(df)
    
         dfc.to_csv('./output_csv/' + varnam+'_at_'+app+'_'+str(iyear)+'-'+str(fyear)+'.csv', index=False)
         # dfc.to_excel('./output_csv/' + varnam+'_at_'+app+'.xlsx')


