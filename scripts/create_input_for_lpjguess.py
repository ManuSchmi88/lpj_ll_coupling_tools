#
# hacky input generator based on the lpjguesstools lgt_createinput suite
#
# TODO:
# refactor into lpjguesstools once abstractions are clear:
# - (1) as second main function to lgt_createinput ?
# - (2) as a submodule in the library similar to plotting


from collections import OrderedDict
import logging
import numpy as np
import glob
import os
import sys

import pandas as pd
import xarray as xr

from lpjguesstools.lgt_createinput._geoprocessing import calc_slope_components, \
                                                         calc_slope, \
                                                         calc_aspect, \
                                                         classify_aspect, \
                                                         classify_landform, \
                                                         calculate_asp_slope, \
                                                         create_tile

from lpjguesstools.lgt_createinput._tpi import calculate_tpi

from lpjguesstools.lgt_createinput.main import define_landform_classes, \
                                               get_tile_summary, \
                                               create_stats_table, \
                                               build_site_netcdf, \
                                               build_landform_netcdf, \
                                               mask_dataset, \
                                               build_compressed, \
                                               create_gridlist

from lpjguesstools.lgt_createinput import _xr_tile

log = logging.getLogger(__name__)

class Bunch(object):
    """Simple data storage class."""
    def __init__(self, adict):
        self.__dict__.update(adict)
    def overwrite(self, adict):
        self.__dict__.update(adict)


def compute_spatial_dataset_landlab(fname_dem, lf_ele_levels):
    """Take a NetCDF file name and return a xarray datasets of dem, slope,
    aspect and water mask layers."""
    
    log.info('Opening file %s ...' % fname_dem)

    dx = 100   # landlab [m], use to be 30
    tpi_radius = 300

    with xr.open_dataset(fname_dem) as src:
        dem = src['topographic__elevation'].squeeze().to_masked_array()
        dem_mask = np.ma.ones(dem.shape)

        dem_filled = dem.copy() # not necessary here, but in for consistency

        Sx, Sy = calc_slope_components(dem_filled, dx)
        slope = calc_slope(Sx, Sy)
        aspect = calc_aspect(Sx, Sy)
        landform = calculate_tpi(dem_filled, slope, tpi_radius, res=dx, TYPE='SIMPLE')

        # check what info and source expect
        ds = create_tile(dem, dem_mask, slope, aspect, landform)

        classify_aspect(ds)
        classify_landform(ds, elevation_levels=lf_ele_levels, TYPE='SIMPLE')
        calculate_asp_slope(ds)

    return ds

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
    
    for file in files_grabbed:
        ds = xr.open_dataset(file)
        attrs = ds.attrs

        if {'lgt.lon', 'lgt.lat', 'lgt.classification'}.issubset( set(attrs.keys() )):
            coordinates.append((attrs['lgt.lat'], attrs['lgt.lon']))
            classifications.append(attrs['lgt.classification'])
            valid_files.append(file)
        else:
            print(f"File {file} does not conform to the format convention.")
            print("Check global attributes")
            continue
    
    if len(set(classifications)) != 1:
        print("Classification attributes differ. Check files.")
        print(classifications)            
        exit(-1)
        
    return (classifications[0].upper(), valid_files, coordinates)
    

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

    print(ds)

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

    LANDLAB_OUTPUT_PATH = os.environ.get('LANDLAB_OUTPUT_PATH', 'landlab/output')

    classification, landlab_files, list_coords = derive_base_info(LANDLAB_OUTPUT_PATH)

    # config object / totally overkill here but kept for consistency
    cfg = Bunch(dict(OUTDIR='.', 
                     CLASSIFICATION=classification, 
                     GRIDLIST_TXT='lpj2ll_gridlist.txt'))

    landlab_files = [extract_variables_from_landlab_ouput(x) for x in landlab_files]

    df_frac, df_elev, df_slope, df_asp_slope, df_aspect = compute_statistics_landlab(landlab_files, list_coords)

    # build netcdfs
    log.info("Building 2D netCDF files")
    dummy_region = [-71.5, -33, -70.5, -26]
    sitenc = build_site_netcdf(SOIL_NC, ELEVATION_NC, extent=dummy_region)
    landformnc = build_landform_netcdf(lf_classes, df_frac, df_elev, df_slope, df_asp_slope, df_aspect, cfg,
                                           lf_ele_levels, refnc=sitenc)

    elev_mask = ~np.ma.getmaskarray(sitenc['elevation'].to_masked_array())
    sand_mask = ~np.ma.getmaskarray(sitenc['sand'].to_masked_array())
    land_mask = ~np.ma.getmaskarray(landformnc['lfcnt'].to_masked_array())
    valid_mask = elev_mask * sand_mask * land_mask

    sitenc = mask_dataset(sitenc, valid_mask)
    landformnc = mask_dataset(landformnc, valid_mask)

    landform_mask = np.where(landformnc['lfcnt'].values == -9999, np.nan, 1)
    #landform_mask = np.where(landform_mask == True, np.nan, 1)
    print(landform_mask)
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

