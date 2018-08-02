#!/usr/bin/env python
#
# create_input_for_lpjguess.py
# ============================
#
# take LandLab output grids and convert them to compressed
# LPJ-GUESS (subpixel) netcdf inputs
#
# Christian Werner (christian.werner@senckenberg.de)
# 2018-08-02

from collections import OrderedDict
import logging
import math
import numpy as np
import glob
import os
import sys

import pandas as pd
import xarray as xr

from lpjguesstools.lgt_createinput.main import define_landform_classes, \
                                               get_tile_summary, \
                                               create_stats_table, \
                                               build_site_netcdf, \
                                               build_landform_netcdf, \
                                               mask_dataset, \
                                               build_compressed, \
                                               create_gridlist

from lpjguesstools.lgt_createinput import _xr_tile
from lpjguesstools.lgt_createinput import _xr_geo

log = logging.getLogger(__name__)

class Bunch(object):
    """Simple data storage class."""
    def __init__(self, adict):
        self.__dict__.update(adict)
    def overwrite(self, adict):
        self.__dict__.update(adict)


def compute_statistics_landlab(list_ds, list_coords):
    
    tiles_stats = []
    for ds, coord in zip(list_ds, list_coords):
        lf_stats = get_tile_summary(ds)     # no cutoff for now
        lf_stats.reset_index(inplace=True)
        number_of_ids = len(lf_stats)
        lat, lon = coord

        coord_tuple = (round(lon,2),round(lat,2), int(number_of_ids))
        lf_stats['coord'] = pd.Series([coord_tuple for _ in range(len(lf_stats))])
        lf_stats.set_index(['coord', 'lf_id'], inplace=True)
        tiles_stats.append( lf_stats )

    df = pd.concat(tiles_stats)

    frac_lf = create_stats_table(df, 'frac_scaled')
    elev_lf = create_stats_table(df, 'elevation')
    slope_lf = create_stats_table(df, 'slope')
    asp_slope_lf = create_stats_table(df, 'asp_slope')
    aspect_lf = create_stats_table(df, 'aspect')
    return (frac_lf, elev_lf, slope_lf, asp_slope_lf, aspect_lf)

def derive_region(coords):
    """Derive bounding box for all coorinates.
    """
    lats, lons = zip(*coords)
    min_lat = math.floor(min(lats))
    max_lat = math.ceil(max(lats))
    min_lon = math.floor(min(lons))
    max_lon = math.ceil(max(lons))
    return [min_lon, min_lat, max_lon, max_lat]


def derive_base_info(ll_inpath):
    """Derive the locations and landform classification
    mode from the landlab grid files"""

    types = ('*.nc', '*.NC')
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(os.path.join(ll_inpath, files)))
    
    # get global attributes (lat, lon, classification)
    # check that classifiaction match
    coordinates = []
    classifications = []
    valid_files = []
    ele_steps = []
    
    for file in files_grabbed:
        ds = xr.open_dataset(file)
        attrs = ds.attrs

        if {'lgt.lon', 'lgt.lat', 'lgt.classification', 'lgt.elevation_step'}.issubset( set(attrs.keys() )):
            coordinates.append((attrs['lgt.lat'], attrs['lgt.lon']))
            classifications.append(attrs['lgt.classification'])
            ele_steps.append(attrs['lgt.elevation_step'])
            
            valid_files.append(file)
        else:
            print(f"File {file} does not conform to the format convention.")
            print("Check global attributes")
            continue
    
    if len(set(classifications)) != 1 or len(set(ele_steps)) != 1:
        print("Classification attributes differ. Check files.")
        print(classifications)
        print(ele_steps)            
        exit(-1)
        
    return (classifications[0].upper(), ele_steps[0], valid_files, coordinates)
    

def extract_variables_from_landlab_ouput(ll_file):
    """Extract 2d data from raw LandLab output and convert to
    lpjguesstool intermediate format. 
    """
    # simple rename
    mapper = {'topographic__elevation' : 'elevation',
              'topographic__steepest_slope': 'slope',
              'tpi__mask': 'mask',
              'aspect' : 'aspect',
              'aspectSlope': 'asp_slope',
              'landform__ID': 'landform_class'
              }

    ds_ll = xr.open_dataset(ll_file)

    # copy data arrays to new file, squeeze, and rename with mapper
    ds = ds_ll.squeeze()[list(mapper.keys())].rename(mapper)
    ds['landform'] = ds_ll.squeeze()['landform__ID'] // 100 % 10  # second last digit
    ds['aspect_class'] = ds_ll.squeeze()['landform__ID'] % 10           # last digit

    return ds

def get_data_location(pkg, resource):
    """Hack to return the data location and not the actual data
    that pkgutil.get_data() returns.
    """
    d = os.path.dirname(sys.modules[pkg].__file__)
    return os.path.join(d, resource)

def main():
    # default soil and elevation data (contained in lpjguesstools package)
    SOIL_NC      = 'GLOBAL_WISESOIL_DOM_05deg.nc'
    ELEVATION_NC = 'GLOBAL_ELEVATION_05deg.nc'
    SOIL_NC = get_data_location("lpjguesstools", "data/"+SOIL_NC)
    ELEVATION_NC = get_data_location("lpjguesstools", "data/"+ELEVATION_NC)

    # get path info for in- and output
    LANDLAB_OUTPUT_PATH = os.environ.get('LANDLAB_OUTPUT_PATH', 'landlab/output')
    LPJGUESS_INPUT_PATH = os.environ.get('LPJGUESS_INPUT_PATH', 'lpjguess/input')

    classification, ele_step, landlab_files, list_coords = derive_base_info(LANDLAB_OUTPUT_PATH)

    lf_classes, lf_ele_levels = define_landform_classes(ele_step, 6000, TYPE=classification)

    # config object / totally overkill here but kept for consistency
    cfg = Bunch(dict(OUTDIR=LPJGUESS_INPUT_PATH, 
                     CLASSIFICATION=classification, 
                     GRIDLIST_TXT='lpj2ll_gridlist.txt'))

    landlab_files = [extract_variables_from_landlab_ouput(x) for x in landlab_files]

    df_frac, df_elev, df_slope, df_asp_slope, df_aspect = compute_statistics_landlab(landlab_files, list_coords)

    # build netcdfs
    log.info("Building 2D netCDF files")

    simulation_domain = derive_region(list_coords)
    sitenc = build_site_netcdf(SOIL_NC, ELEVATION_NC, extent=simulation_domain)
    landformnc = build_landform_netcdf(lf_classes, 
                                       df_frac,
                                       df_elev,
                                       df_slope,
                                       df_asp_slope,
                                       df_aspect,
                                       cfg,
                                       lf_ele_levels, refnc=sitenc)

    elev_mask = ~np.ma.getmaskarray(sitenc['elevation'].to_masked_array())
    sand_mask = ~np.ma.getmaskarray(sitenc['sand'].to_masked_array())
    land_mask = ~np.ma.getmaskarray(landformnc['lfcnt'].to_masked_array())
    valid_mask = elev_mask * sand_mask * land_mask

    sitenc = mask_dataset(sitenc, valid_mask)
    landformnc = mask_dataset(landformnc, valid_mask)

    landform_mask = np.where(landformnc['lfcnt'].values == -9999, np.nan, 1)
    #landform_mask = np.where(landform_mask == True, np.nan, 1)
    for v in sitenc.data_vars:
        sitenc[v][:] = sitenc[v].values * landform_mask


    # write 2d/ 3d netcdf files
    sitenc.to_netcdf(os.path.join(cfg.OUTDIR, 'lpj2ll_sites_2d.nc'),
                     format='NETCDF4_CLASSIC')
    landformnc.to_netcdf(os.path.join(cfg.OUTDIR, 'lpj2ll_landforms_2d.nc'),
                         format='NETCDF4_CLASSIC')

    # convert to compressed netcdf format
    log.info("Building compressed format netCDF files")
    ids_2d, comp_sitenc = build_compressed(sitenc)
    ids_2db, comp_landformnc = build_compressed(landformnc)

    # write netcdf files
    ids_2d.to_netcdf(os.path.join(cfg.OUTDIR, "lpj2ll_land_ids_2d.nc"),
                     format='NETCDF4_CLASSIC')

    comp_landformnc.to_netcdf(os.path.join(cfg.OUTDIR, "lpj2ll_landform_data.nc"),
                              format='NETCDF4_CLASSIC')
    comp_sitenc.to_netcdf(os.path.join(cfg.OUTDIR, "lpj2ll_site_data.nc"),
                          format='NETCDF4_CLASSIC')

    # gridlist file
    log.info("Creating gridlist file")
    gridlist = create_gridlist(ids_2d)
    open(os.path.join(cfg.OUTDIR, cfg.GRIDLIST_TXT), 'w').write(gridlist)

    log.info("Done")

if __name__ == '__main__':
    main()

