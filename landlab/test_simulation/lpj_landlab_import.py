import numpy as np

"""
Script that takes LPJGUESS .csv output and creates
Landlab-style timeseries and maps the values to 
the node-ids within landlab. 

For further information about node-ids, look up
the landformClassifier component in Landlab-ESD-codebase

 
Created by Manuel Schmid, 16.07.2018
"""


#create vegetation timeseries from LPJGUESS DATA
def createVegiTimeseriesFromCsv(csv):
    vd = np.genfromtxt(csv, delimiter = ';', names = True)
    lfArr = []
    for i in vd.dtype.fields:
        lfArr.append(i)
    lfArr = lfArr[1:]
    lfArr = np.array(lfArr).astype(int)
    
    return lfArr, vd

def mapVegetationOnLandform(grid, vegetationData, lfID, timeindex):
    
    #vegCovTimeseries = []
    
    _vegCov = np.zeros(np.shape(grid.at_node['landform__ID']))
    for ids in lfID:
        _vegCov[grid.at_node['landform__ID'] == ids] = vegetationData[str(ids)][timeindex] * 0.01
    
    return _vegCov

def getMAPTimeseriesFromCSV(csv):
    #load .csv file
    pt = np.genfromtxt(csv, delimiter = ';', names = True)
    #create placeholder array to save values in
    _precipArray = []
    for i in range(len(pt)):
        _precipArray.append(pt[i][1] * 0.1)  #read only value from 1 colum, they are all the same anyway
        
    return _precipArray
