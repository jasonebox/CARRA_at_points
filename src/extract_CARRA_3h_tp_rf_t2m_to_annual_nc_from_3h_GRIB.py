#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reads raw 3h t2m mean and accumulated precipitation CARRA data, computes daily totals/averages and outputs these to annual nc files

updated Nov 2022
@author: Jason Box, GEUS, jeb@geus.dk
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

years=np.arange(1991,2020).astype('str')
# years=np.arange(1998,1999).astype('str')
# years=np.arange(2021,2022).astype('str')
# years=np.arange(2017,2018).astype('str')

# for a later version that maps the result
ni=1269 ; nj=1069

path='/Users/jason/Dropbox/CARRA/CARRA_rain/'
raw_path='/Users/jason/0_dat/CARRA_raw/'
# raw_path='/Volumes/LaCie/0_dat/CARRA/CARRA_raw/'
raw_path='/Volumes/Samsung_T5/0_dat/CARRA/CARRA_raw/'
outpath='/Users/jason/0_dat/CARRA_at_TIN_points/3h/' 
# outpath='/Volumes/LaCie/0_dat/CARRA/output/annual/' 

os.chdir(path)


fn='./ancil/2.5km_CARRA_west_lat_1269x1069.npy'
lat=np.fromfile(fn, dtype=np.float32)
lat=lat.reshape(ni, nj)

fn='./ancil/2.5km_CARRA_west_lon_1269x1069.npy'
lon=np.fromfile(fn, dtype=np.float32)
lon=lon.reshape(ni, nj)


#functions to read in files
def gett2m(val, var,year,suffix):
    fn=raw_path+'CARRA-West_T2m_mean_'+year+'_'+val+'h.'+suffix
    print(fn)
    ds=xr.open_dataset(fn,engine='cfgrib')
    t2m=ds.variables[var].values#-273.15
    time=ds.variables['time'].values#-273.15
    return t2m,time

def gettp(val, var,year,suffix):
    fn=raw_path+'CARRA-West_prec_'+year+'_3h_acc_'+val+'h.'+suffix
    print(fn)
    ds=xr.open_dataset(fn,engine='cfgrib')
    tp=ds.variables[var].values 
    return tp

def get_rf(t2m,tp):
    # rain phasing
    rainos=0.
    x0=0.5 ; x1=2.5
    # x0=-2 ; x1=0
    x0-=rainos
    x1-=rainos
    y0=0 ; y1=1
    a1=(y1-y0)/(x1-x0)
    a0=y0-a1*x0

    f=np.zeros((ni,nj))
    # np.squeeze(t2m, axis=0)
    v=np.where(((t2m>x0)&(t2m<x1)))
    f[v]=t2m[v]*a1+a0
    v=np.where(t2m>x1) ; f[v]=1
    v=np.where(t2m<x0) ; f[v]=0
    
    rf=tp*f
    return rf

# # plot function
# def plt_x(var,lo,hi,nam,cbarnam,units):
#     plt.close()
#     plt.imshow(var,vmin=lo,vmax=hi)
#     plt.title(timestamp_string)
#     plt.axis('off')
#     clb = plt.colorbar(fraction=0.046/2., pad=0.08)
#     clb.ax.set_title(units,fontsize=7)
#     ly='p'
#     fig_path='/Users/jason/0_dat/CARRA_temp/'
#     if ly=='p':
#         plt.savefig(fig_path+'_'+nam+'_'+timestamp_string+'.png', bbox_inches='tight', dpi=200)#, facecolor=bg, edgecolor=fg)

def write_nc(varnam,varxx,n_days,ni,nj,outpath):

    ofile=outpath+varnam+'_'+year+'.nc'
    
    print("making .nc file for "+varnam)
    os.system("/bin/rm "+ofile)
    ncfile = Dataset(ofile,mode='w',format='NETCDF4_CLASSIC')
    lat_dim = ncfile.createDimension('lat', nj)     # latitude axis
    lon_dim = ncfile.createDimension('lon', ni)    # longitude axis
    time_dim = ncfile.createDimension('time', n_days) # unlimited axis (can be appended to)
    
    # for dim in ncfile.dimensions.items():
    #     print(dim)            
    # ncfile.title=varnam+' '+stat_type
    ncfile.subtitle="subtitle"
    # print(ncfile.subtitle)
    # print(ncfile)
    
    lat = ncfile.createVariable('lat', np.float32, ('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lon = ncfile.createVariable('lon', np.float32, ('lon',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'
    time = ncfile.createVariable('time', np.float64, ('time',))
    # time.units = 'days since '+year+'-01-01'
    time.units = 'days'
    time.long_name = 'time'
    # Define a 3D variable to hold the data
    print("compressing")
    temp = ncfile.createVariable(varnam,np.float32,('time','lon','lat'),zlib=True,least_significant_digit=3) # note: unlimited dimension is leftmost
    temp.units = 'mm/day' # degrees Kelvin
    temp.standard_name = varnam # this is a CF standard name
    # print(temp)
    
    nlats = len(lat_dim); nlons = len(lon_dim); ntimes = 3
    # Write latitudes, longitudes.
    # Note: the ":" is necessary in these "write" statements
    # lat[:] = -90. + (180./nlats)*np.arange(nlats) # south pole to north pole
    # lon[:] = (180./nlats)*np.arange(nlons) # Greenwich meridian eastward
    # create a 3D array of random numbers
    # data_arr = np.random.uniform(low=280,high=330,size=(ntimes,nlats,nlons))
    # Write the data.  This writes the whole 3D netCDF variable all at once.
    # temp[:,:,:] = np.rot90(result,2)  # Appends data along unlimited dimension
    temp[:,:,:] = varxx  # Appends data along unlimited dimension

    # temp2 = ncfile.createVariable("confidence",np.float64,('time','lat','lon')) # note: unlimited dimension is leftmost
    # temp2.units = "unitless" # degrees Kelvin
    # temp2.standard_name = "confidence" # this is a CF standard name
    # temp2[:,:,:] = confidence  # Appends data along unlimited dimension

    # print("-- Wrote data, temp.shape is now ", temp.shape)
    print(ofile)
    # read data back from variable (by slicing it), print min and max
    # print("-- Min/Max values:", temp[:,:,:].min(), temp[:,:,:].max())
    
    ncfile.close(); print('Dataset is closed!')
    return

fn='./site_coords/TIN_Greenland_ccordinates_for_CARRA.csv'
df=pd.read_csv(fn)
n=len(df)

# ---------- read in data
time_list_t2m=['00_UTC_fl_3-6','00_UTC_fl_6-9','00_UTC_fl_9-12','00_UTC_fl_12-15','12_UTC_fl_3-6','12_UTC_fl_6-9','12_UTC_fl_9-12','12_UTC_fl_12-15']
time_list_tp =['00_UTC_fl_6-3','00_UTC_fl_9-6','00_UTC_fl_12-9','00_UTC_fl_15-12','12_UTC_fl_6-3','12_UTC_fl_9-6','12_UTC_fl_12-9','12_UTC_fl_15-12']

time_list_hours=['03','06','09','12','15','18','21','00']

for yy,year in enumerate(years):

        if calendar.isleap(int(year)):
            print(year+' is leap year')
            n_days=366
        else:
            print(year+' is not a leap year')
            n_days=365

        tp_3h_4dcube=np.zeros((((8,n_days,ni,nj))))
        rf_3h_4dcube=np.zeros((((8,n_days,ni,nj))))
        t2m_3h_4dcube=np.zeros((((8,n_days,ni,nj))))

   
        var=['t2m', 'mn2t6', 'tp', 'time']
        
        for cc in np.arange(0,8):
        # for cc in np.arange(0,1):
        # for cc in np.arange(6,8):
            print('cc',cc)
            t_varnam=var[0]
            if cc==0:t_varnam=var[1]
            if cc==4:t_varnam=var[1]
            # if str(year)==2021:
            suffix='grb2'
            # else:suffix='nc'
            t2m,time=gett2m(time_list_t2m[cc], t_varnam,year,suffix) 
            # if str(year)==2021:
            suffix='grib'
            # else:suffix='nc'            
            tp=gettp(time_list_tp[cc], var[2],year,suffix)
            tp_3h_4dcube[cc,:,:,:]=tp
            t2m_3h_4dcube[cc,:,:,:]=t2m
            # time=gett2m(time_list_t2m[cc], var[3])
            if cc==0:
                timex=pd.to_datetime(time).strftime('%Y-%m-%d-%H')
                jday=pd.to_datetime(time).strftime('%j')
#%%
        tp=0 ; t2m=0

        size=4

        tp_3h_5dcube=np.zeros(((((3,8,n_days,9,9)))))
        rf_3h_5dcube=np.zeros(((((3,8,n_days,9,9)))))
        t2m_3h_5dcube=np.zeros(((((3,8,n_days,9,9)))))     
        
        
        for dd in range(n_days):
        # for dd in range(2):
            # if dd==190:
            if dd!=1900:
                # print(dd)
                # rf_daily=np.zeros((ni,nj))
                # t2m_daily=np.zeros((ni,nj))
                # tp_daily=np.zeros((ni,nj))
                print(year,dd)
                for hh in range(8):            
                # ------------------------------------------- rain fraction
                    t2m=t2m_3h_4dcube[hh,dd,:,:]-273.15
                    rfx=get_rf(t2m,tp_3h_4dcube[hh,dd,:,:]) ; rf_rot_T=np.rot90(rfx.T)
                    # rf_daily+=np.rot90(rfx.T)
                    t2m_rot_T=np.rot90(t2m.T)
                    # t2m_daily+=t2m_rot_T
                    tpx=np.rot90(tp_3h_4dcube[hh,dd,:,:].T)
                    # tp_daily+=np.rot90(tp_3h_4dcube[hh,dd,:,:].T)
                    
                    for k in range(0,n):
                        tp_3h_5dcube[k,hh,dd,:,:]=tpx[df['j'][k]-size:df['j'][k]+size+1,df['i'][k]-size:df['i'][k]+size+1]
                        rf_3h_5dcube[k,hh,dd,:,:]=rf_rot_T[df['j'][k]-size:df['j'][k]+size+1,df['i'][k]-size:df['i'][k]+size+1]
                        t2m_3h_5dcube[k,hh,dd,:,:]=t2m_rot_T[df['j'][k]-size:df['j'][k]+size+1,df['i'][k]-size:df['i'][k]+size+1]
                        
                    # plt.imshow(rf_rot_T,vmin=0,vmax=10)
                    # plt.imshow(t2m_rot_T)
                    # plt.grid(False)
                    # plt.colorbar()
                    # plt.title(hh)
                    # plt.show()
                    
        print('saving '+year)
        with open('./3h/'+year+'.npy', 'wb') as f:
            np.save(f, rf_3h_5dcube)
            np.save(f, tp_3h_5dcube)
            np.save(f, t2m_3h_5dcube)
        
#%% check
        # with open('./3h/'+year+'.npy', 'rb') as f:
        #     rf_3h_5dcube=np.load(f)
        #     tp_3h_5dcube=np.load(f)
        #     t2m_3h_5dcube=np.load(f)

        # location=1 # sisimiut_1
        # print(np.shape(rf_3h_5dcube))
        # sumx=np.zeros((np.shape(t2m_3h_5dcube)[3],np.shape(rf_3h_5dcube)[4]))
        # # for dd in range(np.shape(rf_3h_5dcube)[2]):
        # for dd in range(95,99):
        #     for hh in range(8):
        #         sumx=t2m_3h_5dcube[location,hh,dd,:,:]

        #         # sumx+=rf_3h_5dcube[location,hh,dd,:,:] # totaling precip
        #         plt.close()
        #         plt.imshow(sumx)
        #         plt.colorbar()
        #         plt.title(df.name[location]+' t2m\nday of year '+str(dd+1)+' '+str((hh+1)*3)+'h UTC')
        #         # plt.axis('off')
        #         ly='x'
        #         if ly == 'x':
        #             plt.show()


