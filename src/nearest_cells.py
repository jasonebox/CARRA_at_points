#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 08:08:56 2020

@author: Jason Box, GEUS, jeb@geus.dk



"""
ly='x'

import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os
import numpy as np
from osgeo import gdal, gdalconst
from netCDF4 import Dataset as NetCDFFile 
import pandas as pd

ni=1269
nj=1069


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


path='/Users/jason/Dropbox/CARRA/CARRA_at_points/'
os.chdir(path)

fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
lat=np.fromfile(fn, dtype=np.float32)
lat=lat.reshape(ni, nj)
lat=np.rot90(lat.T)

fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
lon=np.fromfile(fn, dtype=np.float32)
lon=lon.reshape(ni, nj)
lon-=360
lon=np.rot90(lon.T)

fn='./ancil/2.5km_CARRA_west_elev_1269x1069.npy'
elev=np.fromfile(fn, dtype=np.float32)
elev=elev.reshape(ni, nj)
elev=np.rot90(elev.T)

#%%
from netCDF4 import Dataset

fn='/Users/jason/Dropbox/CARRA/ancil/Const.Clim.sfx.nc'
nc = Dataset(fn, mode='r')
# print(nc.variables)
z = nc.variables[variables[i]][0,0,:,:]
#%%

fn='/Users/jason/Dropbox/CARRA/CARRA_at_points/site_coords/Thomas_Ingemann_Nielsen.csv'
df=pd.read_csv(fn)

# n=len(df)
# for i in range(0,n):        
#     dist=haversine_np(df.lon[i],df.lat[i], lon, lat)
#     v=np.where(dist==np.min(dist))
#     print(df.name[i],lat[v],lon[v],dist[v])

central_points_i=[]
central_points_j=[]

n=len(df)
for k in range(0,n):
# for k in range(1):
    min_dist=900
    central_i=0
    central_j=0
    for i in range(0,ni):
        for j in range(0,nj):
            dist=haversine_np(df.lon[k],df.lat[k], lon[i,j], lat[i,j])
            if dist<min_dist:
                min_dist=dist
                print(k,min_dist,i,j)
                if min_dist<2:
                    central_i=i
                    central_j=j
                    central_points_j.append(j)
                    central_points_i.append(i)
            #         break
            #     else:
            #         continue
            #     break
            # else:
            #     continue            # v=np.where(dist==np.min(dist))
    # central_points_j.append
            # print(df.name[i],lat[v],lon[v],dist[v])

#%%
print(lat[central_points_i,central_points_j],lon[central_points_i,central_points_j])

df['j']=pd.Series(np.array(central_points_i))
df['i']=pd.Series(np.array(central_points_j))

df.to_csv('/Users/jason/Dropbox/CARRA/CARRA_at_points/site_coords/TIN_Greenland_ccordinates_for_CARRA.csv',index=None)
#%%
x=elev.copy()
size=4
for k in range(0,n):
    x[df['j'][k]-size:df['j'][k]+size,df['i'][k]-size:df['i'][k]+size]=3000
plt.imshow(x)

#%%
fn='/Users/jason/Dropbox/CARRA/CARRA_at_points/site_coords/TIN_Greenland_ccordinates_for_CARRA.csv'
df=pd.read_csv(fn)
n=len(df)

for k in range(0,n):
    with open('/Users/jason/0_dat/CARRA_at_TIN_points/lat_lon_elev_'+df.name[k]+'.npy', 'wb') as f:
        np.save(f, lat[df['j'][k]-size:df['j'][k]+size+1,df['i'][k]-size:df['i'][k]+size+1])
        np.save(f, lon[df['j'][k]-size:df['j'][k]+size+1,df['i'][k]-size:df['i'][k]+size+1])
        np.save(f, elev[df['j'][k]-size:df['j'][k]+size+1,df['i'][k]-size:df['i'][k]+size+1])

#%% test
with open('/Users/jason/0_dat/CARRA_at_TIN_points/lat_lon_elev.npy', 'rb') as f:
    lat = np.load(f)
    lon = np.load(f)
    elev = np.load(f)

plt.imshow(elev)
plt.colorbar()