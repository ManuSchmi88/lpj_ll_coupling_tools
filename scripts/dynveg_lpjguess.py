from enum import Enum
from landlab import Component
import logging
import numpy as np
import os
import xarray as xr
import shutil
import sys
from tqdm import tqdm
from typing import List, Optional


logPath = '.'
fileName = 'dynveg_lpjguess'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(logPath, fileName)),
        logging.StreamHandler()
    ])

log = logging.getLogger()


class TS(Enum):
    DAILY = 1
    MONTHLY = 2

def split_climate(ds_files:List[str], 
                  dt:int, 
                  ds_path:Optional[str]=None, 
                  dest_path:Optional[str]=None, 
                  time_step:TS=TS.MONTHLY) -> None:
    """Split climte files into dt-length chunks"""

    for ds_file in ds_files:
        fpath = os.path.join(ds_path, ds_file) if ds_path else ds_file
 
        with xr.open_dataset(fpath, decode_times=False) as ds:
            n_episodes = len(ds.time) // (dt*12)
            log.debug('Number of climate episodes: %d' % n_episodes)
            if time_step == TS.MONTHLY:
                episode = np.repeat(list(range(n_episodes)), dt*12)
            else:
                episode = np.repeat(list(range(n_episodes)), dt*365)
            ds['grouper'] = xr.DataArray(episode, coords=[('time', ds.time.values)])
            log.info('Splitting file %s' % ds_file)
            for g_cnt, ds_grp in tqdm(ds.groupby(ds.grouper)):
                del ds_grp['grouper']
                ds_grp.to_netcdf('%s_%s.nc' % (fpath.replace('.nc',''), str(g_cnt).zfill(6)), format='NETCDF4_CLASSIC')
            
def prepare_filestructure(dest:str, source:Optional[str]=None) -> None:
    SOURCE = 'default_sources'

    if os.path.isdir(dest): shutil.rmtree(dest)
    if source:
        shutil.copytree(source, dest)        
    else:
        shutil.copytree(SOURCE, dest)
    os.makedirs(os.path.join(dest, 'lfdata'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'climdata'), exist_ok=True)

def prepare_input(dest:str) -> None:

    prepare_filestructure(dest)

    # move this to a config or make it smarter
    vars = ['prec', 'temp', 'rad']
    ds_files = ['egu2018_%s_35ka_def_landid.nc' % v for v in vars]
    split_climate(ds_files, dt=100, ds_path='../forcings/climdata', dest_path=dest, time_step=TS.MONTHLY)


class DynVeg_LpjGuess(Component):
    """classify a DEM in different landform, according to slope, elevation and aspect"""

    @property
    def spinup(self):
        return self._spinup
    
    @property
    def timestep(self):
        return self._current_timestep

    def __init__(self, dest:str, spinup:bool = False):
        self._spinup = spinup
        self._current_timestep = 0

        prepare_input(dest)

    def run_one_step(self) -> None:
        pass



def test_dynveg_contructor():
    c = DynVeg_LpjGuess('../lpjguess/input2')
    print(c.spinup)
    print(c)

if __name__ == '__main__':
    log.info('Starting dynveg lpjguess component')
    test_dynveg_contructor()

