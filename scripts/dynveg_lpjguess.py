from enum import Enum
from landlab import Component
import logging
import numpy as np
import os
import xarray as xr
import shutil
from string import Template
import sys
import time
from tqdm import tqdm
from typing import Dict, List, Optional

LPJGUESS_INPUT_PATH = os.environ.get('LPJGUESS_INPUT_PATH', 'lpjguess/input')
LPJGUESS_TEMPLATE_PATH = os.environ.get('LPJGUESS_TEMPLATE_PATH', 'lpjguess.template')
LPJGUESS_FORCINGS_PATH = os.environ.get('LPJGUESS_FORCINGS_PATH', 'forcings')
LPJGUESS_INS_FILE_TPL = os.environ.get('LPJGUESS_INS_FILE_TPL', 'lpjguess.ins.tpl')

logPath = '.'
fileName = 'dynveg_lpjguess'

logging.basicConfig(
    level=logging.INFO,
    format = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s",
    #format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(logPath, fileName)),
        logging.StreamHandler()
    ])

log = logging.getLogger()


class TS(Enum):
    DAILY = 1
    MONTHLY = 2

def execute_lpjguess(dest:str) -> None:
    '''Run LPJ-Guess for one time-step'''
    log.info('RUN LPJ-GUESS')

def fill_template(template: str, data: Dict[str, str]) -> str:
    """Fill template file with specific data from dict"""
    with open( template, 'rU' ) as f:
        src = Template( f.read() )
    return src.substitute(data)

def split_climate(ds_files:List[str], 
                  dt:int, 
                  ds_path:Optional[str]=None, 
                  dest_path:Optional[str]=None, 
                  time_step:TS=TS.MONTHLY) -> None:
    """Split climte files into dt-length chunks"""
    log.debug('ds_path: %s' % ds_path)
    log.debug('dest_path: %s' % dest_path)
    log.debug(ds_files)

    for ds_file in ds_files:
        fpath = os.path.join(ds_path, ds_file) if ds_path else ds_file
        log.debug(fpath)

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
                foutname = os.path.basename(fpath.replace('.nc',''))
                foutname = os.path.join(dest_path, '%s_%s.nc' % (foutname, str(g_cnt).zfill(6)))
                ds_grp.to_netcdf(foutname, format='NETCDF4_CLASSIC')
            
def prepare_filestructure(dest:str, source:Optional[str]=None) -> None:
    log.debug('dest: %s' % dest)
    if os.path.isdir(dest):
        log.warn('destination folder exists... removing in 3 sec')
        time.sleep(3)
        shutil.rmtree(dest)
    if source:
        shutil.copytree(source, dest)        
    else:
        shutil.copytree(LPJGUESS_TEMPLATE_PATH, dest)
    os.makedirs(os.path.join(dest, 'lfdata'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'climdata'), exist_ok=True)

def prepare_input(dest:str) -> None:
    log.debug('dest: %s' % dest)
    prepare_filestructure(dest)

    # move this to a config or make it smarter
    vars = ['prec', 'temp', 'rad']
    ds_files = ['egu2018_%s_35ka_def_landid.nc' % v for v in vars]
    split_climate(ds_files, dt=100, ds_path=os.path.join(LPJGUESS_FORCINGS_PATH, 'climdata'),
                                    dest_path=os.path.join(LPJGUESS_INPUT_PATH, 'climdata'), 
                                    time_step=TS.MONTHLY)

def prepare_runfiles(dest:str, dt:int) -> None:
    """Prepare files specific to this dt run"""
    # fill template files with per-run data:
    run_data = {# climate data
                'CLIMPREC': 'sample_run_clim_prec_00000.nc',
                'CLIMWET':  'sample_run_clim_prec_00000.nc',
                'CLIMRAD':  'sample_run_clim_rad_00000.nc',
                'CLIMTEMP': 'sample_run_clim_temp_00000.nc',
                # landform files
                'LFDATA': 'lf_data.nc',
                'SITEDATA': 'site_data.nc',
                # setup data
                'GRIDLIST': 'gridlist.txt',
                'NYEARSPINUP': '500',
                'RESTART': '0'
                }

    insfile = fill_template( os.path.join(dest, LPJGUESS_INS_FILE_TPL), run_data )
    open(os.path.join(dest, 'lpjguess.ins'), 'w').write(insfile)

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
        self._dest = dest
        prepare_input(self._dest)

    def run_one_step(self) -> None:
        prepare_runfiles(self._dest, self._current_timestep)
        execute_lpjguess(self._dest)


def test_dynveg_contructor():

    c = DynVeg_LpjGuess(LPJGUESS_INPUT_PATH)
    print(c.spinup)
    print(c)

if __name__ == '__main__':
    log.info('Starting dynveg lpjguess component')
    test_dynveg_contructor()

