"""
Script which runs once through all the netcdf files and creates a serialized pickle. 
This way you only need to go through all the files once.

This script needs to sit in some sort of sub-folder of your main model-directory since it
looks for the path "../NC" and goes through all the netcdf files in there.

created by: Manuel Schmid, 24.07.2018
"""

from netCDF4 import Dataset
import numpy as np
import os 
import glob
import time
import pickle
import matplotlib.pyplot as plt

print('module import finished.')

netcdf_string = '../NC/' #path to netcdf-output
#initialize the dictionary which will hold all the data
data_strings = ['topographic__elevation',
                'topographic__steepest_slope',
                'landform__ID',
                'soil_production__rate',
                'soil__depth',
                'vegetation__density',
                'rainvalue',
                'topographic_position__index',
                'topographic_position__class',
                'drainage_area',
                'sediment__flux']

#populate data_dictionary
data_dictionary = {}
for names in data_strings:
    data_dictionary[names] = []

print('populated dictionary. starting with netcdf loading.')

#for checking progress
counter = 0
files = next(os.walk('../NC'))
files = len(files[2])

for filename in sorted(glob.glob(os.path.join(netcdf_string, '*.nc')), key=os.path.getmtime):
    #dump the current netcdf_file in a object
    data_dump = Dataset(filename)
    for data_field in data_strings:
        #create a temporary data container for each data-field and delete boundary nodes
        _container = data_dump.variables[data_field][:]
        _container = _container[0]
        _container = np.delete(_container,  0, axis = 0)
        _container = np.delete(_container, -1, axis = 0)
        _container = np.delete(_container,  0, axis = 1)
        _container = np.delete(_container, -1, axis = 1)
        
        #re-locate everything into the data_dictionary
        data_dictionary[data_field].append(_container)
        
    counter += 1
    if counter % 500 == 0:
        print('500 files done - {} to go'.format(str(files - counter)))
    
print('Done with File loading. - starting serialization, this may take a while')
pickle.dump( data_dictionary, open( "data_dictionary.p", "wb" ) )
print('Finished creating pickle-object named: "data_dictionary.p"')
