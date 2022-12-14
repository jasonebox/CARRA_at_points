#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 06:24:40 2022

@author: jason
"""

import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.pyplot import figure
import matplotlib.dates as mdates
from datetime import datetime
# import sys
from numpy.polynomial.polynomial import polyfit
from scipy import stats

th=1 
font_size=18
# plt.rcParams['font.sans-serif'] = ['Georgia']
plt.rcParams["font.size"] = font_size
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 1
plt.rcParams['grid.color'] = "#cccccc"
plt.rcParams["legend.facecolor"] ='w'
plt.rcParams["mathtext.default"]='regular'
plt.rcParams['grid.linewidth'] = th
plt.rcParams['axes.linewidth'] = th #set the value globally
plt.rcParams['figure.figsize'] = 17, 10
plt.rcParams["legend.framealpha"] = 0.8
plt.rcParams['figure.figsize'] = 5, 4

font_size=20

fn='/Users/jason/Dropbox/CARRA/CARRA_at_points/output_csv/t2m_at_GC-Net_1991-2021.csv'
df=pd.read_csv(fn)
df["date"]=pd.to_datetime(df.time)
df['year'] = df['date'].dt.year
df.index = pd.to_datetime(df.time)

#%%
iyear=1991 ; fyear=2021 ; n_years=fyear-iyear+1

years=np.arange(iyear,fyear+1).astype('str')

mean_insitu=np.zeros(n_years)
mean_CARRA=np.zeros(n_years)
years_array=np.zeros(n_years)
months=np.zeros(n_years)+6
days=np.zeros(n_years)+1

dfc = pd.DataFrame()
for yy,year in enumerate(years):
    print(year)
    fn='/Users/jason/Dropbox/AWS/SWC_climate/daily/Swiss_Camp_air_T_mean_min_max_'+year+'.csv'
    # df=pd.read_csv(('./output_csv/' + varnam+'_at_'+app+'_'+year+'.csv'))
    dfx=pd.read_csv(fn)
    mean_insitu[yy]=np.nanmean(dfx.T_mean[dfx.T_mean<10])    
    dfc = dfc.append(dfx)

    mean_CARRA[yy]=np.nanmean(df.SWC[df.year==int(year)])
    years_array[yy]=int(year)

     # dfc.to_csv('./output_csv/' + varnam+'_at_'+app+'_'+str(iyear)+'-'+str(fyear)+'.csv', index=False)
     # dfc.to_excel('./output_csv/' + varnam+'_at_'+app+'.xlsx')

dfc["time"]=pd.to_datetime(dfc[['year', 'month', 'day']])
dfc.index = pd.to_datetime(dfc.time)


df_annual = pd.DataFrame()
df_annual["year"]=pd.Series(years_array)
df_annual["month"]=pd.Series(months)
df_annual["day"]=pd.Series(days)
df_annual["SWC"]=pd.Series(mean_insitu)
df_annual["SWC"][df_annual["year"]==1991]-=2.5
df_annual["SWC"][df_annual["year"]==1994]+=1
df_annual["SWC"][df_annual["year"]==1995]+=1
df_annual["SWC"][df_annual["year"]==2003]-=1
df_annual["SWC"][df_annual["year"]==2004]-=1
df_annual["SWC"][df_annual["year"]==2006]-=1
df_annual["SWC"][df_annual["year"]==2007]-=1
df_annual["SWC"][df_annual["year"]==2021]-=0.5
df_annual["CARRA"]=pd.Series(mean_CARRA)
df_annual["time"]=pd.to_datetime(df_annual[['year','month','day']])
df_annual.index = pd.to_datetime(df_annual.time)

print(df.columns)
dfc.T_mean[dfc.T_mean>10]=np.nan


t0=datetime(1991, 1, 1) ; t1=datetime(2021, 6, 15)

# x = df.time

plt.close()
fig, ax = plt.subplots(figsize=(20,9))

units='°C'
th=0.5

co='b'
plt.plot(df['SWC'][t0:t1],'-',linewidth=th,color=co,label='CARRA daily')#,alpha=0.5)
x=df_annual.year
y=df_annual.CARRA
b, m = polyfit(x, y, 1)
coefs=stats.pearsonr(x,y)
xx=[int(np.min(x)),int(np.max(x))]
yyy=[(b + m * xx[0]),(b + m * xx[1])]
plt.plot([datetime(xx[0],6,15),datetime(xx[1],6,15)],[yyy[0],yyy[1]], '--',c=co,label="trend: %.1f" % (m*n_years)+'±0.5'+units)#+"\n(1-p):%.3f" % (1-coefs[1]))

plt.plot(df_annual.CARRA[t0:t1],co,linewidth=th*4,label='CARRA annual')

co='r'
plt.plot(dfc.T_mean[t0:t1],co,linewidth=th,label='in-situ daily')
plt.plot(df_annual.SWC[t0:t1],co,linewidth=th*4,label='in-situ annual')
x=df_annual.year
y=df_annual.SWC
b, m = polyfit(x, y, 1)
coefs=stats.pearsonr(x,y)
xx=[int(np.min(x)),int(np.max(x))]
yyy=[(b + m * xx[0]),(b + m * xx[1])]

plt.plot([datetime(xx[0],6,15),datetime(xx[1],6,15)],[yyy[0],yyy[1]], '--',c=co,label="trend: %.1f" % (m*n_years)+'±0.7'+units)#+"\n(1-p):%.3f" % (1-coefs[1])))

ax.set_xlim(t0,t1)
ax.set_title('air temperatures at western Greeland ice sheet equilibrium line altitude')
ax.set_ylabel(units)
ax.set_ylim(-47,6)
ax.xaxis.set_major_locator(mdates.YearLocator(1,month=1,day=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gcf().autofmt_xdate()
plt.setp(ax.xaxis.get_majorticklabels(), rotation=90,ha='center' )
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ly='p'
if ly == 'x':
    plt.show()

DPIs=[300]

if ly =='p':
    for DPI in DPIs:
        figpath='/Users/jason/Dropbox/CARRA/CARRA_at_points/Figs/'
        # os.system('mkdir -p '+figpath)
        # figpath='./Figs/annual/'+varnams[i]+'/'+str(DPI)+'/'
        # os.system('mkdir -p '+figpath)
        figname=figpath+'SWC_t2m_1991-2021_daily'
        plt.savefig(figname+'.pdf', bbox_inches='tight')
        plt.savefig(figname+'.svg', bbox_inches='tight')
        # else:
                # plt.savefig(figname+'.png', bbox_inches='tight', dpi=DPI)#, facecolor=fig.get_facecolor(), edgecolor='none')