#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon NOV 28 19:16:42 2022

updated NOV 2022
@author: Jason Box, GEUS, jeb@geus.dk
"""
# import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import calendar
from numpy.polynomial.polynomial import polyfit
from scipy import stats

# global plot settings
font_size=16
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

varname=['t2m'] ; app='PROMICE'
varname=['t2m'] ; app='TAS'
varname=['t2m'] ; app='t-10m'
# varname=['rf'] ; app='PROMICE'
varname=['t2m','tp'] ; app='points'
# varname=['t2m'] ; app='Zackenberg'
# varname=['rf'] ; app='Zackenberg'
varname=['tp','rf','t2m'] ; app='Zackenberg'
# varname=['tp','rf','t2m'] ; app='GNET'
varname=['tp','rf','t2m'] ; app='GEUS_df2'
app='GC-Net'


# varname=['tp','rf','t2m'] ; app='Zackenberg'

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

if app=='GEUS_df2':
    # read PROMICE locations
    meta = pd.read_csv('/Users/jason/Dropbox/AWS/_merged/GEUS_df2_sites_20221128.csv')   
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

# #%% combine multi variables into a single dataframe
# varname=['tp','rf','t2m']
# fn='/Users/jason/Dropbox/CARRA/CARRA_rain/output_csv/'+varname[0]+'_at_'+app+'.csv'
# df0=pd.read_csv(fn)
# df0=df0.rename(columns={app: 'tp'})

# fn='/Users/jason/Dropbox/CARRA/CARRA_rain/output_csv/'+varname[1]+'_at_'+app+'.csv'
# df1=pd.read_csv(fn)
# df1=df1.rename(columns={app: 'rf'}) 

# df0 = pd.merge(df1,df0, how='left', on="time")

# for varnam in varname[2:3]:
#       print(varnam)
#       fn='/Users/jason/Dropbox/CARRA/CARRA_rain/output_csv/'+varnam+'_at_'+app+'.csv'
#       df=pd.read_csv(fn)
#       df=df.rename(columns={app: varnam})
#       df = pd.merge(df0,df, how='left', on="time")

# print(df)
# df.to_csv('./output_csv/CARRA_tp_rf_t2m_at_'+app+'.csv', index=False)
# # df.to_excel('./output_csv/CARRA_tp_rf_t2m_at_'+app+'.xlsx')
# #%% plot daily for years

# df=pd.read_csv('/Users/jason/Dropbox/CARRA/CARRA_rain/output_csv/CARRA_tp_rf_t2m_at_'+app+'.csv')
# df["time"]=pd.to_datetime(df['time'])
# df.index = pd.to_datetime(df.time)

# fig, ax = plt.subplots(figsize=(10,10))

# # plt.plot(df.rf)
# # plt.plot(df.tp-df.rf)
# plt.plot(df.t2m)
# # plt.plot(df.tp-df.rf)

# plt.setp(ax.xaxis.get_majorticklabels(), rotation=90,ha='center' )
# # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %Hh'))
# plt.title(app+' CARRA rainfall')
# plt.ylabel('mm')
# plt.legend()

#%%

n_years=fyear-iyear+1

varname=['tp','rf','t2m']
varname2=['total precipitation','rainfall','2m air temperature']

# for vv,varnam in enumerate(varname[0:2]):
for vv,varnam in enumerate(varname):

    df=pd.read_csv('./output_csv/' + varnam+'_at_'+app+'_'+str(iyear)+'-'+str(fyear)+'.csv')
    df["time"]=pd.to_datetime(df['time'])
    # df.index = pd.to_datetime(df.time)
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day

    N=len(meta.station_name)
    for ss,site in enumerate(meta.station_name):
        # if site=='SWC_O':
        # if site=='SDM':
        # if site=='QAS_L':
        # if site=='NAU':
        # if site=='Zackenberg':
        if site!='null':
            print(varnam,site,N-ss,meta.lat[ss],meta.lon[ss])#,meta.elev[ss])
    
    
            months=np.zeros(n_years*12)   
            years=np.arange(iyear,fyear+1)
            tot=np.zeros(n_years)
            monthly_stat=np.zeros((n_years,12))
            monthly_min=np.zeros((n_years,12))
            monthly_max=np.zeros((n_years,12))
            monthly_std=np.zeros((n_years,12))
            ann_stat=np.zeros(n_years)
        
            units='mm'
            # if vv==0:
            if vv>=2:
                units='°C'
                # cc=0
            for yy,year in enumerate(years):
                v=df.year==year
                # print(site,N-ss,year)
                if vv<2:
                    tot[yy]=np.sum(df[site][v])
                    prec='%.1f'
                    statnam='total'
                else:
                    tot[yy]=np.mean(df[site][v])
                    prec='%.1f'
                    statnam='mean'
                for mm in range(12):
                    v=((df.year==year)&(df.month==mm+1))
                    # months[cc]=mm+1
                    if vv<2:
                        result=np.sum(df[site][v])
                    else:
                        result=np.mean(df[site][v])
                    monthly_stat[yy,mm]=result
                    monthly_min[yy,mm]=np.min(df[site][v])
                    if vv<2:
                        if monthly_min[yy,mm]<=0:monthly_min[yy,mm]=0
                    monthly_max[yy,mm]=np.max(df[site][v])
                    monthly_std[yy,mm]=np.std(df[site][v])
                    # print(year,mm,result)
                v=(df.year==year)
                if vv<2:
                    ann_stat[yy]=np.sum(df[site][v])
                else:
                    ann_stat[yy]=np.mean(df[site][v])     
            
            df2 = pd.DataFrame(columns = ['year','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANN']) 
            df2.index.name = 'index'
            df2["year"]=pd.Series(np.arange(iyear,fyear+1))
            months=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
            for mm,mon in enumerate(months):
                df2[mon]=pd.Series(monthly_stat[:,mm])
                df2[mon] = df2[mon].apply(lambda x: prec % x)
            
            df2['DJF']=np.nan
            df2['MAM']=np.nan
            df2['JJA']=np.nan
            df2['SON']=np.nan

            for i in range(len(df2)):
                if i>0:
                    # if ( (~np.isnan(df2.JAN[i])) & (~np.isnan(df2.FEB[i]))  & (~np.isnan(df2.DEC[i-1])) ):
                    df2.DJF[i]=np.mean([float(df2.JAN[i]),float(df2.FEB[i]),float(df2.DEC[i-1])])
                # if ( (~np.isnan(df2.MAR[i])) & (~np.isnan(df2.APR[i]))  & (~np.isnan(df2.MAY[i])) ):
                df2.MAM[i]=np.mean([float(df2.MAR[i]),float(df2.APR[i]),float(df2.MAY[i])])
                # if ( (~np.isnan(df2.JUN[i])) & (~np.isnan(df2.JUL[i]))  & (~np.isnan(df2.AUG[i])) ):
                df2.JJA[i]=np.mean([float(df2.JUN[i]),float(df2.JUL[i]),float(df2.AUG[i])])
                # if ( (~np.isnan(df2.SEP[i])) & (~np.isnan(df2.OCT[i]))  & (~np.isnan(df2.NOV[i])) ):
                df2.SON[i]=np.mean([float(df2.SEP[i]),float(df2.OCT[i]),float(df2.NOV[i])])
                # print(df2.Year[i],df2.SON[i])

            df2['ANN']=pd.Series(ann_stat)
            df2['ANN'] = df2['ANN'].apply(lambda x: prec % x)
            df2['DJF'] = df2['DJF'].map(lambda x: '%.1f' % x)
            df2['MAM'] = df2['MAM'].map(lambda x: '%.1f' % x)
            df2['JJA'] = df2['JJA'].map(lambda x: '%.1f' % x)
            df2['SON'] = df2['SON'].map(lambda x: '%.1f' % x)

            ofile='/Users/jason/Dropbox/CARRA/CARRA_at_points/monthly/'+app+'_'+varnam+'_'+site+'_'+str(iyear)+'-'+str(fyear)+'_'+statnam
            df2.to_csv(ofile+'.csv',index=None)
    
            df2 = pd.DataFrame(columns = ['year','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']) 
            df2.index.name = 'index'
            df2["year"]=pd.Series(np.arange(iyear,fyear+1))
            for mm,mon in enumerate(months):
                df2[mon]=pd.Series(monthly_min[:,mm])
                df2[mon] = df2[mon].apply(lambda x: prec % x)
            ofile='/Users/jason/Dropbox/CARRA/CARRA_at_points/monthly/'+app+'_'+varnam+'_'+site+'_'+str(iyear)+'-'+str(fyear)+'_min'
            df2.to_csv(ofile+'.csv',index=None)
    
            df2 = pd.DataFrame(columns = ['year','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']) 
            df2.index.name = 'index'
            df2["year"]=pd.Series(np.arange(iyear,fyear+1))
            for mm,mon in enumerate(months):
                df2[mon]=pd.Series(monthly_max[:,mm])
                df2[mon] = df2[mon].apply(lambda x: prec % x)
            ofile='/Users/jason/Dropbox/CARRA/CARRA_at_points/monthly/'+app+'_'+varnam+'_'+site+'_'+str(iyear)+'-'+str(fyear)+'_max'
            df2.to_csv(ofile+'.csv',index=None)
    
            df2 = pd.DataFrame(columns = ['year','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']) 
            df2.index.name = 'index'
            df2["year"]=pd.Series(np.arange(iyear,fyear+1))
            for mm,mon in enumerate(months):
                df2[mon]=pd.Series(monthly_std[:,mm])
                df2[mon] = df2[mon].apply(lambda x: '%.1f' % x)
            ofile='/Users/jason/Dropbox/CARRA/CARRA_at_points/monthly/'+app+'_'+varnam+'_'+site+'_'+str(iyear)+'-'+str(fyear)+'_std'
            df2.to_csv(ofile+'.csv',index=None)
            
            ly='p'
            plt.close()
            x=years ;y=tot
            plt.plot(x,y,label=varnam)
            b, m = polyfit(x, y, 1)
            coefs=stats.pearsonr(x,y)
            xx=[np.min(x),np.max(x)]
            yyy=[(b + m * xx[0]),(b + m * xx[1])]
            plt.plot(xx,yyy, '-',c='grey',label="trend: %.1f" % (m*n_years)+' '+units+"\n(1-p):%.3f" % (1-coefs[1]))

            plt.title(site+' CARRA annual '+varname2[vv])
            plt.legend()
            plt.ylabel(units)
            plt.xlabel('year')
            

            if ly == 'x':
                plt.show()


            if ly =='p':
                figpath='/Users/jason/Dropbox/CARRA/CARRA_at_points/monthly/'
                # os.system('mkdir -p '+figpath)
                # figpath='./Figs/annual/'+varnams[i]+'/'+str(DPI)+'/'
                # os.system('mkdir -p '+figpath)
                figname=figpath+app+'_'+varnam+'_'+site+'_'+str(iyear)+'-'+str(fyear)+'_'+statnam
                # if i<2:
                plt.savefig(figname+'.pdf', bbox_inches='tight')

    ##%% output daily dataframes
    # data=[[years],[means],[stds]]
    
        # df1d = pd.DataFrame(columns = ['year', 'day of year',varnam,'min','max','stdev'])
        # df1d.index.name = 'index'
        # df1d["year"]=pd.Series(years)
        # df1d["month"]=pd.Series(months)
        # df1d[varnam]=pd.Series(monthly_stat)
        # df1d['min']=pd.Series(monthly_min)
        # df1d['max']=pd.Series(monthly_max)
        # df1d['stdev']=pd.Series(monthly_std)
        # # df1d['t1 mean'] = df1d['t1 mean'].apply(lambda x: '%.2f' % x)
        # # df1d["t1 count"]=pd.Series(t1count)
        # # df1d["t2 mean"]=pd.Series(t2mean)
        # # df1d['t2 mean'] = df1d['t2 mean'].apply(lambda x: '%.2f' % x)
        # # df1d["t2 count"]=pd.Series(t2count)
        # ofile='/Users/jason/Dropbox/CARRA/CARRA_at_points/monthly/'+app+'_monthly_'+str(iyear)+'-'+str(fyear)
        # # df1d.to_csv(ofile+'.csv')
        # df1d.to_csv(ofile+'_tower.csv')
        # # df1d.to_excel(ofile+'.xlsx')
    

