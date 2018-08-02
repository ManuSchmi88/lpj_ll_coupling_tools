#!/usr/bin/env python
#
# climdata_for_synthetic_climate_cycles_03_split.py
# =================================================
#
# split climdata into 100yr packages
#
# Christian Werner (christian.werner@senckenberg.de)
# 2018-08-02

import xarray as xr

# short cycle
# 3000yr - 500yr - 3000yr cycle

# long cycle
# 3000yr - 2000yr - 3000yr cycle 

def add_time_attrs(ds, calendar_year=0):
    ds['time'].attrs['units'] = "days since 1-1-15 00:00:00" ;
    ds['time'].attrs['axis'] = "T" ;
    ds['time'].attrs['long_name'] = "time" ;
    ds['time'].attrs['standard_name'] = "time" ;
    ds['time'].attrs['calendar'] = f"{calendar_year} yr B.P." ;

batch_size = 100 # years

for var in ['prec', 'rad', 'temp']:
    print(f"{var}...")

    simtypes =['def', 'scy', 'lcy'] 

    files = [f"egu2018_{var}_35ka_{simtype}_landid.nc" for simtype in simtypes]

    for file in files:
        ds = xr.open_dataset(file, decode_times=False)

        # this is messy, we actually use a negative BP string here (check if LPJ is happy)
        # indicating the calendar is actually shifted to the future
        for icnt, i in enumerate(range(0, len(ds['time']), len(ds['time'])//(12*batch_size))):
            ds_ = ds.isel(time=slice(i, i+(12*batch_size)))
            add_time_attrs(ds_, -i)
            ds_.to_netcdf(f"{file[:-3]}.{str(icnt).zfill(4)}.nc", format='NETCDF4_CLASSIC')
     