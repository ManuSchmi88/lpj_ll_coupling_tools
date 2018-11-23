import glob
import xarray as xr
import numpy as np

files = sorted(glob.glob('egu2018*_landid.nc'))

for file in files:
    #print(file)
    nc = xr.open_dataset(file, decode_times=False)
    nc_0 = nc.copy(deep=True)
    nc_1 = nc.copy(deep=True)
    
    for v in nc.data_vars:
        nc_1[v][0][:] = nc_1[v][1]
        nc_1[v][1][:] = np.nan 
        
        nc_0[v][0][:] = np.nan 

    nc_0.to_netcdf(f'{file[:-3]}-0.nc')
    nc_1.to_netcdf(f'{file[:-3]}-1.nc')
     