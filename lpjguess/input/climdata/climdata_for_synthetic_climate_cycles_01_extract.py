#!/usr/bin/env python
#
# climdata_for_synthetic_climate_cycles_01_extract.py
# ===================================================
#
# Christian Werner (christian.werner@senckenberg.de)
# 2018-08-02
#  
# description:
# - extract last 100 yr climate intervals from TraCE-21ka transient timeseries
# - currently picks two EarthShapes sites and maps to 0,1 ids
# - def: 0=pan de azucar, 1=la campana; lender: 0=st gracia, 1=nahuelbuta 
#
# output:
# two netcdf files (def, lender) that can be combined at will to create egu2018
# cycles

import xarray as xr

for var in ['prec', 'rad', 'temp']:
    # compose baseline climfiles
    ds = xr.open_dataset("TraCE_biascorrected.0-220.focussites_%s_landid.nc" % var, 
                      decode_times=False)
    print(ds)

    # x: original, y: lender
    x = ds.sel(land_id=[207, 355])
    y = ds.sel(land_id=[283, 490])

    x['land_id'] = [0,1]
    y['land_id'] = [0,1]
 
    x.to_netcdf("egu2018_%s_landid.nc" % var)

    # restrict to last 100 years
    t = x.time.values
    t = t[-12*100:]
    
    # 100yr default
    x_100 = x.sel(time=slice(t[0],t[-1]))
    x_100.to_netcdf("egu2018_%s_100yr_landid.nc" % var)

    # 100yr from adjacent site
    y_100 = y.sel(time=slice(t[0],t[-1]))
    y_100.to_netcdf("egu2018_%s_100yr_lender_landid.nc" % var)

    # mix a
    
