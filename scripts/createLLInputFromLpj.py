import numpy as np
import xarray as xr

"""
set of scripts which makes post-processed lpj-output landlab compatible
"""

def read_csv_files(filename):
    """
    reads in the csv files from lpj

    resulting 2D-arrays have format:
        
        tree_fpc[0] = landform__ID
        tree_fpc[1] = according vegetation cover
    """
    
    fpc_array = np.genfromtxt(filename, delimiter = ';', names = True)

    return fpc_array

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
    
    for landform in fpc_array[0]:
        fpc_grid[grid.at_node['landform__ID'] == landform] = fpc_array[str(landform)]

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


def run_one_step(grid, treefile, shrubfile, grassfile, method = 'cumulative') :
    
    #read different files
    tree_fpc = read_csv_files(treefile)
    shrub_fpc = read_csv_files(shrubfile)
    grass_fpc = read_csv_files(grassfile)

    if method == 'cumulative':
        #calculate cumulative 
        cum_fpc = calc_cumulative_fpc(tree_fpc, grass_fpc, shrub_fpc) 
        #map on landlab grid
        mg.at_node['vegetation__density'] = map_fpc_per_landform_on_grid(grid, cum_fpc)

    
    if method == 'individual':
        #add individual landlab fields to the grid
        grid.add_zeros('grass_fpc')
        grid.add_zeros('tree_fpc')
        grid.add_zeros('shrub_fpc')
        
        #map values to individual fiels 
        mg.at_noda['grass_fpc'] = map_fpc_per_landform_on_grid(grid, grass_fpc)
        mg.at_node['tree_fpc'] = map_fpc_per_landform_on_grid(grid, tree_fpc)
        mg.at_node['shrub_fpc'] = map_fpc_per_landform_on_grid(grid, shrub_fpc)
     
    
