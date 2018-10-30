from enum import Enum
import glob
from landlab import Component
import logging
import numpy as np
import os
import xarray as xr
import shutil
from string import Template
import subprocess
import sys
import time
from tqdm import tqdm
from typing import Dict, List, Optional

from create_input_for_lpjguess import main as create_input_main

# define consts - source environemt.sh
LPJGUESS_INPUT_PATH = os.environ.get('LPJGUESS_INPUT_PATH', 'run')
LPJGUESS_TEMPLATE_PATH = os.environ.get('LPJGUESS_TEMPLATE_PATH', 'lpjguess.template')
LPJGUESS_FORCINGS_PATH = os.environ.get('LPJGUESS_FORCINGS_PATH', 'forcings')
LPJGUESS_INS_FILE_TPL = os.environ.get('LPJGUESS_INS_FILE_TPL', 'lpjguess.ins.tpl')
LPJGUESS_BIN = os.environ.get('LPJGUESS_BIN', 'guess')
LPJGUESS_CO2FILE = os.environ.get('LPJGUESS_CO2FILE', 'co2.txt') 


# logging setup
logPath = '.'
fileName = 'dynveg_lpjguess'

logging.basicConfig(
    level=logging.DEBUG,
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

def add_time_attrs(ds, calendar_year=0):
    ds['time'].attrs['units'] = "days since 1-1-15 00:00:00" ;
    ds['time'].attrs['axis'] = "T" ;
    ds['time'].attrs['long_name'] = "time" ;
    ds['time'].attrs['standard_name'] = "time" ;
    ds['time'].attrs['calendar'] = "%d yr B.P." % calendar_year


def generate_landform_files(dest:str) -> None:
    log.info('Convert landlab netcdf data to lfdata fromat')
    create_input_main()


def execute_lpjguess(dest:str) -> None:
    '''Run LPJ-Guess for one time-step'''
    log.info('Execute LPJ-Guess run')

    p = subprocess.Popen([LPJGUESS_BIN, '-input', 'sp', 'lpjguess.ins'], cwd=dest)
    p.wait()

def move_state(dest:str) -> None:
    '''Move state dumpm files into loaddir for next timestep'''
    log.info('Move state to loaddir')
    state_files = glob.glob(os.path.join(dest, 'dumpdir_eor/*'))
    for state_file in state_files:
        shutil.copy(state_file, os.path.join(dest, 'loaddir'))


def fill_template(template: str, data: Dict[str, str]) -> str:
    """Fill template file with specific data from dict"""
    log.debug('Fill LPJ-GUESS ins template')
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

                # modify time coord
                # us first dt years data
                if g_cnt == 0:
                    time_ = ds_grp['time'][:dt*12]

                add_time_attrs(ds, calendar_year=22_000)
                foutname = os.path.basename(fpath.replace('.nc',''))
                foutname = os.path.join(dest_path, '%s_%s.nc' % (foutname, str(g_cnt).zfill(6)))
                ds_grp.to_netcdf(foutname, format='NETCDF4_CLASSIC')
        
    # copy co2 file
    src = os.path.join(ds_path, LPJGUESS_CO2FILE) if ds_path else LPJGUESS_CO2FILE
    log.debug('co2_path: %s' % ds_path) 
    shutil.copyfile(src, os.path.join(dest_path, LPJGUESS_CO2FILE))
            

            
def prepare_filestructure(dest:str, source:Optional[str]=None) -> None:
    log.debug('Prepare file structure')
    log.debug('Dest: %s' % dest)
    if os.path.isdir(dest):
        log.warn('Destination folder exists... removing in 3 sec')
        time.sleep(3)
        shutil.rmtree(dest)
    if source:
        shutil.copytree(source, dest)        
    else:
        shutil.copytree(LPJGUESS_TEMPLATE_PATH, dest)
    os.makedirs(os.path.join(dest, 'input', 'lfdata'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'input', 'climdata'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'output'), exist_ok=True)


def prepare_input(dest:str) -> None:
    log.debug('Prepare input')
    log.debug('dest: %s' % dest)
    
    prepare_filestructure(dest)

    # move this to a config or make it smarter
    vars = ['prec', 'temp', 'rad']
    ds_files = ['egu2018_%s_35ka_def_landid.nc' % v for v in vars]
    split_climate(ds_files, dt=100, ds_path=os.path.join(LPJGUESS_FORCINGS_PATH, 'climdata'),
                                    dest_path=os.path.join(LPJGUESS_INPUT_PATH, 'input', 'climdata'), 
                                    time_step=TS.MONTHLY)

def prepare_runfiles(dest:str, dt:int) -> None:
    """Prepare files specific to this dt run"""
    # fill template files with per-run data:
    restart = '0' if dt == 0 else '1'

    run_data = {# climate data
                'CLIMPREC': 'egu2018_prec_35ka_def_landid_%s.nc' % str(dt).zfill(6),
                'CLIMWET':  'egu2018_prec_35ka_def_landid_%s.nc' % str(dt).zfill(6),
                'CLIMRAD':  'egu2018_rad_35ka_def_landid_%s.nc' % str(dt).zfill(6),
                'CLIMTEMP': 'egu2018_temp_35ka_def_landid_%s.nc' % str(dt).zfill(6),
                # landform files
                'LFDATA': 'lpj2ll_landform_data.nc',
                'SITEDATA': 'lpj2ll_site_data.nc',
                # setup data
                'GRIDLIST': 'landid.txt',
                'NYEARSPINUP': '500',
                'RESTART': restart
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

    def __init__(self, dest:str):
        self._spinup = True
        self._current_timestep = 0
        self._dest = dest
        prepare_input(self._dest)

    def run_one_step(self) -> None:
        prepare_runfiles(self._dest, self._current_timestep)
        generate_landform_files(self._dest)
        execute_lpjguess(self._dest)
        move_state(self._dest)
        if self.timestep == 0:
            self._spinup = False
        self._current_timestep += 1


def test_dynveg_contructor():

    c = DynVeg_LpjGuess(LPJGUESS_INPUT_PATH)
    print(c.spinup)
    print(c)

    for i in range(3):
        c.run_one_step()

if __name__ == '__main__':
    log.info('Starting dynveg lpjguess component')
    test_dynveg_contructor()

