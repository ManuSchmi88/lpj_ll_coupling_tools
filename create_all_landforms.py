def create_all_landforms(uplift, runtime, elevation_bin_size, grid):
    """
    create an array of all possible landform_ID's, as a function of uplift rate and total model runtime
    and map them on the closed boundary nodes of the grid. 
    
    There are 4 TPI Classes:
        1 = hilltop
        3 = mid slope
        4 = flat surface
        6 = valley

    There are 5 Aspect Classes:
        1 = N
        2 = E
        3 = S 
        4 = W
        0 = FLAT SURFACE (with TPI = 1, 4, 6)

    So we need to create a list with all possible permuations ranking from 
    min_ele_ID to max_ele_ID + TPI + ASPECT

    """

    import numpy as np

    landform_list = []
    possible_slope_aspect_ids = ['10', '60', '40', '31', '32', '33', '34']
    
    _min_initial_elevation = np.min(grid.at_node['topographic__elevation'])
    _max_possible_elevation_gain = uplift * runtime #note: have to be same units
    _max_possible_elevation = _max_possible_elevation_gain + _min_initial_elevation
    
    _max_possible_ele_id = int(_max_possible_elevation / elevation_bin_size)
    _min_possible_ele_id = int(_min_initial_elevation / elevation_bin_size)
    
    #create a list with all possible elevation_ids within the grid
    for ele in range(_min_possible_ele_id, _max_possible_ele_id):
        for j in possible_slope_aspect_ids:
            _lf = str(ele) + j
            landform_list.append(_lf)
            
    for lf, ind in zip(landform_list, range(len(landform_list))):
        grid.at_node['landform__ID'][ind] = lf