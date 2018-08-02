#!/usr/bin/env python
#
# climdata_for_synthetic_climate_cycles_02_combine.py
# ===================================================
#
# merge climdata subset to arbitray cycles
#
# Christian Werner (christian.werner@senckenberg.de)
# 2018-08-02

import xarray as xr
import numpy as np
# short cycle
# 3000yr - 500yr - 3000yr cycle

# long cycle
# 3000yr - 2000yr - 3000yr cycle 

def add_time_attrs(ds):
    ds['time'].attrs['units'] = "days since 1-1-15 00:00:00" ;
    ds['time'].attrs['axis'] = "T" ;
    ds['time'].attrs['long_name'] = "time" ;
    ds['time'].attrs['standard_name'] = "time" ;
    ds['time'].attrs['calendar'] = "0 yr B.P." ;


for var in ['prec', 'rad', 'temp']:
    # compose baseline climfiles
    ds = xr.open_dataset("egu2018_%s_100yr_landid.nc" % var, 
                      decode_times=False)
    
    ds2 = xr.open_dataset("egu2018_%s_100yr_landid.nc" % var, 
                      decode_times=False)
    # cycle 1
    # continuous
    x1 = xr.concat([ds]*350, dim='time')
    x1['time'] = np.cumsum(np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]*(350*100)) - 1)
    add_time_attrs(x1)
    x1.to_netcdf('egu2018_%s_35ka_def_landid.nc' % var, format='NETCDF4_CLASSIC')
    
    # cycle 2 (short)
    a = [ds]*30
    b = [ds2] * 5
    c = a+b
    ts = c*10
    x2 = xr.concat(ts, dim='time')
    x2['time'] = np.cumsum(np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]*(350*100)) - 1)
    add_time_attrs(x2)
    x2.to_netcdf('egu2018_%s_35ka_scy_landid.nc' % var, format='NETCDF4_CLASSIC')
    
    
    # cycle 3 (long)
    a = [ds]*30
    b = [ds2] * 20
    c = a+b
    ts = c*7
    x3 = xr.concat(ts, dim='time')
    x3['time'] = np.cumsum(np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]*(350*100)) - 1)
    add_time_attrs(x3)
    x3.to_netcdf('egu2018_%s_35ka_lcy_landid.nc' % var, format='NETCDF4_CLASSIC')
     