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
import os
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

def main():

    # default soil and elevation data (contained in package)
    import pkg_resources
    SOIL_NC      = 'GLOBAL_WISESOIL_DOM_05deg.nc'
    ELEVATION_NC = 'GLOBAL_ELEVATION_05deg.nc'

    list_ds_landlab = []
    list_coords = []

    # config object / totally overkill here but kept for consistency
    cfg = Bunch(dict(OUTDIR='.', 
                     CLASSIFICATION='SIMPLE', 
                     GRIDLIST_TXT='lpj2ll_gridlist.txt'))

    for landlab_dem_file in ['10Perc_SS_Topo_rot90.nc', '70Perc_SS_Topo_rot90.nc']:

        if '10Perc_SS' in landlab_dem_file:
            tile_avg = 259.33
            lat = -26.25
            lon = -70.75
        elif '70Perc_SS' in landlab_dem_file:
            tile_avg = 401.46
            lat = -32.75
            lon = -71.25

        lf_classes, lf_ele_levels = define_landform_classes(200, 6000, TYPE=cfg.CLASSIFICATION)

        ds_landlab = compute_spatial_dataset_landlab(landlab_dem_file, lf_ele_levels)
        list_ds_landlab.append( ds_landlab )
        list_coords.append( (lat,lon) )
        
        # write files to compare with manu
        ds_landlab.to_netcdf(landlab_dem_file[:-3] + '_info.nc', format='NETCDF4_CLASSIC')
    
    df_frac, df_elev, df_slope, df_asp_slope, df_aspect = compute_statistics_landlab(list_ds_landlab, list_coords)

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

