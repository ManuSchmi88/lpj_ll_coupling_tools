def load_landlab_nc_files(folder_list, name_list, do_pickle = False, pickle_name = 'data_pickle.p'):
    """
    function that takes all the paths in the provided input list and extracts the netcdf
    files into a dictionary. 
    Requirement for this to work is the "standart-manu-landlab-folder-structure" so, 
    /model_dir/ll_output/NC and the provided input paths are the full paths to the 
    base model_dir. 
    The second input is a list of names that are used as dictionary indizes.
    
    if argument pickle is set to True, the function automatically creates a serialized
    object in the folder where it was run which encompasses the data_dictionary and is named
    data_pickle.p
    
    Extracted Parameters are: 
        -Topographic Elevation
        -Topographic Slope
        -Erosion Rate
        -Vegetation Density (either cumulative or individual)
        -Soil Depth
        -Soil Production Rate
    
    input:
        folder_list - type: list, eg. ['path1', 'path2', ...]
        name_list   = type: list, eg. ['name1', 'name2', ...]
        do_pickle   = boolean, if True then pickle is used for saving a binary file with the data_object
        pickle_name = type: str, eg. 'MyData.p'
        
    output:
        data_dict   - simu1
                    - simu2
                    - ...
        if do_pickle == True, this saves a binary file in the runtime folder
                    
                    
    created by: 
        Manuel Schmid, 09.3.2019
    """
    
    #import the needed modules
    import numpy as np
    import os, glob, pickle
    from netCDF4 import Dataset
    
    
    #first check if folder_list is really a list
    if isinstance(folder_list, list) and isinstance(name_list, list):
        pass
    else:
        raise ValueError('Provided input must be of type list')
        
    #create dictionary
    data_dict = {}
    
    #create parameter list
    parameters = [
    "topographic__elevation",
    "topographic__steepest_slope",
    "soil__depth",
    "sediment__flux",
    "landform__ID",
    "precipitation",
    "erosion__rate",
    "tree_fpc",
    "grass_fpc",
    "shrub_fpc",
    "vegetation__density",
    "fluvial_erodibility__soil",
    "fluvial_erodibility__bedrock"
    ]

    #create last part of path to /ll_output/NC
    nc_path = 'll_output/NC/'
    
    #populate dictionary with names from name_list
    for name in name_list:
        data_dict[name] = {}
        for p in parameters:
            data_dict[name][p] = []
    
    #data unpacking
    for folder, name in zip(folder_list, name_list):
        #check if last / was provided with path and create full_path string
        if folder[-1] == '/':
            full_path = os.path.join(folder, nc_path)
        else:
            full_path = os.path.join(folder, '/', nc_path)
        
        counter = 0
        
        print('Data loading of ' + str(name) + ' Simulation')
        
        for nc_file in sorted(glob.glob(os.path.join(full_path, "*.nc")), key = os.path.getmtime):
            _dataDump = Dataset(nc_file)
            counter += 1
            #print('DEBUG: - For loop activated')
            if counter % 100 == 0:
                print('100 files done')
                
            #check if the simulation was run with individual fpcs or cumulative
            if 'tree_fpc' and 'shrub_fpc' and 'grass_fpc' in _dataDump.variables:

                #adjust parameters list to only use the plant_fpc values
                if 'vegetation__density' in parameters:
                    parameters.remove('vegetation__density')
                else:
                    pass
                
                for p in parameters:
                    _cutDump = _dataDump.variables[p][:][0]
                    #delete boundary nodes
                    _cutDump = np.delete(_cutDump, 0 , axis = 0) 
                    _cutDump = np.delete(_cutDump,-1 , axis = 0)
                    _cutDump = np.delete(_cutDump, 0 , axis = 1)
                    _cutDump = np.delete(_cutDump,-1 , axis = 1)

                    data_dict[name][p].append(np.mean(_cutDump)) 
                    
            else:
                
                #adjust paramaters list to just use "vegetation__density"
                if 'tree_fpc' and 'shrub_fpc' and 'grass_fpc' in parameters:
                    parameters.remove('tree_fpc')
                    paramaters.remove('shrub_fpc')
                    paramaters.remove('grass_fpc')
                else:
                    pass
                
                for p in parameters:
                    _cutDump = _dataDump.variables[p][:][0]
                    #delete boundary nodes
                    _cutDump = np.delete(_cutDump, 0 , axis = 0) 
                    _cutDump = np.delete(_cutDump,-1 , axis = 0)
                    _cutDump = np.delete(_cutDump, 0 , axis = 1)
                    _cutDump = np.delete(_cutDump,-1 , axis = 1)

                    data_dict[name][p].append(np.mean(_cutDump))
    
    if do_pickle == True:
        print('do_pickle was choosen. Starting serialization...')
        pickle.dump( data_dict, open( pickle_name, "wb" ))
        print('file saved in: ' + str(pickle_name))
    else: 
        print('do_pickle is false. I am not saving a binary file!')
        pass
    
    #returns the data_dictionary
    return data_dict