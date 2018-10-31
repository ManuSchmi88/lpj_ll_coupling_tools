import numpy as np
import xarray as xr
import pandas as pd

"""
set of scripts which makes post-processed lpj-output landlab compatible
"""

import logging, time
from timer import timed

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

def _calc_fpc(lai):
    """Calculate FPC using the LPJ-GUESS method

    """
    return (1.0 - np.exp(-0.5 * lai)) * 100

def read_csv_files(filename, ftype='lai', pft_class='total'):
    """
    reads in the out files from lpj and convertes to aggregated values
    
    sp_lai.out

    resulting 2D-arrays have format:
        
        tree_fpc[0] = landform__ID
        tree_fpc[1] = according vegetation cover
    """

    if ftype != 'lai':
        raise NotImplementedError

    # these are custom column names (can be configures in LPJ ins file!)
    index_cols = ['Lat', 'Lon', 'Year', 'Stand', 'Patch'] 
    tree_cols = ['TeBE_tm','TeBE_itm','TeBE_itscl','TeBS_itm','TeNE','BBS_itm','BBE_itm']
    shrub_cols = ['BE_s','TeR_s','TeE_s']
    grass_cols = ['C3G']
    total_col = ['Total']

    if pft_class == 'total':
        requested_cols = total_col
    elif pft_class == 'grass':
        requested_cols = grass_cols
    elif pft_class == 'shrub':
        requested_cols = shrub_cols
    elif pft_class == 'tree':
        requested_cols = tree_cols
    else:
        raise NotImplementedError

    df = pd.read_table(filename, delim_whitespace=True)[index_cols + requested_cols]
    df = df[df.Stand > 0]
    del df['Patch']
    df_grp = df.groupby(['Lon', 'Lat', 'Year', 'Stand'], sort = False).mean()
    df_grp = df_grp.apply(_calc_fpc, 1).sum(axis=1)
    x = df_grp.reset_index().set_index(['Year', 'Stand'])
    del x['Lon'], x['Lat']
    fpc_array = x.mean(level=1).T
    #fpc_array.to_csv(f'fpc_{v}.csv', index=False)
    #fpc_array = np.genfromtxt(filename, delimiter = ';', names = True)

    #'little' hacky right now. This returns a recarray object created
    #from pandas-dataframe.
    return fpc_array.to_records()

def map_fpc_per_landform_on_grid(grid, fpc_array):
    """
    extract the tree fractional cover per landform
    
    assumes that the landlab grid object which is passed already
    has a data field 'landform__id' which is used to create
    numpy arrays with correct dimensions for mapping vegetation
    data
    """

    #creates grid_structure for landlab
    fpc_grid = np.zeros(np.shape(grid.at_node['landform__ID']))
    
    for landform in fpc_array.dtype.names[1:]:
        fpc_grid[grid.at_node['landform__ID'] == int(landform)] = fpc_array[str(landform)]

    #print('map_fpc_per_landform_on_grid was run')
    return fpc_grid


def calc_cumulative_fpc(tree_fpc, grass_fpc, shrub_fpc):
    """
    If you want to use total vegetation cover instead of individual cover, this
    script adds up trees, shrubs, grass
    """

    total_fpc = tree_fpc[1] + shrub_fpc[1] + grass_fpc[1]

    cumulative_fpc = tree_fpc.copy()
    cumulative_fpc[1] = total_fpc


    return cumulative_fpc

#@timed(logger) 
def lpj_import_run_one_step(grid, inputFile, method = 'cumulative'):
    """
    main function for input_conversion to be called from landlab driver file
    """
    
    if method == 'cumulative':
        cum_fpc = read_csv_files(inputFile, ftype = 'lai', pft_class = 'total')
        grid.at_node['vegetation__density'] = map_fpc_per_landform_on_grid(grid, cum_fpc)

    elif method == 'individual':
        #add individual landlab fields to the grid if they don't exist
        if all(fpc in grid.keys('node') for fpc in ('grass_fpc', 'tree_fpc',
            'shrub_fpc')):
            #no need to initalize something. just do nothing
            pass
        else:
            grid.add_zeros('node', 'grass_fpc')
            grid.add_zeros('node', 'tree_fpc')
            grid.add_zeros('node', 'shrub_fpc')

        grass_fpc = read_csv_files(inputFile, ftype = 'lai', pft_class = 'grass')
        tree_fpc = read_csv_files(inputFile, ftype = 'lai', pft_class = 'tree')
        shrub_fpc = read_csv_files(inputFile, ftype = 'lai', pft_class = 'shrub')

        #map values to individual fiels
        grid.at_node['grass_fpc'] = map_fpc_per_landform_on_grid(grid, grass_fpc)
        grid.at_node['tree_fpc'] = map_fpc_per_landform_on_grid(grid, tree_fpc)
        grid.at_node['shrub_fpc'] = map_fpc_per_landform_on_grid(grid, shrub_fpc)

    else: 
        raise NotImplementedError
