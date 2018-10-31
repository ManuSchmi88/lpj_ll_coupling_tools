import coloredlogs
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

# this is a bit hacky - import scripts main as a function and use here
# should be refactored into this script
from create_input_for_lpjguess import main as create_input_main

# define consts - source environemt.sh
#LPJGUESS_INPUT_PATH = os.environ.get('LPJGUESS_INPUT_PATH', 'run')
#LPJGUESS_TEMPLATE_PATH = os.environ.get('LPJGUESS_TEMPLATE_PATH', 'lpjguess.template')
#LPJGUESS_FORCINGS_PATH = os.environ.get('LPJGUESS_FORCINGS_PATH', 'forcings')
#LPJGUESS_INS_FILE_TPL = os.environ.get('LPJGUESS_INS_FILE_TPL', 'lpjguess.ins.tpl')
#LPJGUESS_BIN = os.environ.get('LPJGUESS_BIN', 'guess')
#LPJGUESS_CO2FILE = os.environ.get('LPJGUESS_CO2FILE', 'co2.txt') 
#


##DIRTY
LPJGUESS_INPUT_PATH = './temp_lpj'
LPJGUESS_TEMPLATE_PATH = './lpjguess.template'
LPJGUESS_FORCINGS_PATH = './forcings'
LPJGUESS_INS_FILE_TPL = 'lpjguess.ins.tpl'
LPJGUESS_BIN = '/esd/esd01/data/mschmid/coupling/build/guess'
LPJGUESS_CO2FILE = 'co2_TraCE_egu2018_35ka_const180ppm.txt'
# logging setup
logPath = '.'
fileName = 'dynveg_lpjguess'

FORMAT="%(levelname).1s %(asctime)s %(filename)s:%(lineno)s - %(funcName).15s :: %(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format = FORMAT,
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(logPath, fileName)),
        logging.StreamHandler()
    ])

log = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', fmt=FORMAT, datefmt="%H:%M:%S")

class TS(Enum):
    DAILY = 1
    MONTHLY = 2

def add_time_attrs(ds, calendar_year=0):
    ds['time'].attrs['units'] = "days since 1-1-15 00:00:00" 
    ds['time'].attrs['axis'] = "T" 
    ds['time'].attrs['long_name'] = "time" 
    ds['time'].attrs['standard_name'] = "time" 
    ds['time'].attrs['calendar'] = "%d yr B.P." % calendar_year


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

                add_time_attrs(ds_grp, calendar_year=22_000)
                foutname = os.path.basename(fpath.replace('.nc',''))
                foutname = os.path.join(dest_path, '%s_%s.nc' % (foutname, str(g_cnt).zfill(6)))
                ds_grp.to_netcdf(foutname, format='NETCDF4_CLASSIC')
        
    # copy co2 file
    src = os.path.join(ds_path, LPJGUESS_CO2FILE) if ds_path else LPJGUESS_CO2FILE
    log.debug('co2_path: %s' % ds_path) 
    shutil.copyfile(src, os.path.join(dest_path, LPJGUESS_CO2FILE))
            

def generate_landform_files(self) -> None:
    log.info('Convert landlab netcdf data to lfdata fromat')
    create_input_main()

def execute_lpjguess(self) -> None:
    '''Run LPJ-Guess for one time-step'''
    log.info('Execute LPJ-Guess run')
    p = subprocess.Popen([LPJGUESS_BIN, '-input', 'sp', 'lpjguess.ins'], cwd=self._dest)
    p.wait()

def move_state(self) -> None:
    '''Move state dumpm files into loaddir for next timestep'''
    log.info('Move state to loaddir')
    state_files = glob.glob(os.path.join(self._dest, 'dumpdir_eor/*'))
    for state_file in state_files:
        shutil.copy(state_file, os.path.join(self._dest, 'loaddir'))

        # for debugging:
        shutil.copy(state_file, os.path.join('tmp.state'))
            
def prepare_filestructure(dest:str, source:Optional[str]=None) -> None:
    log.debug('Prepare file structure')
    log.debug('Dest: %s' % dest)
    if os.path.isdir(dest):
        log.fatal('Destination folder exists...')
        exit(-1)
        #time.sleep(3)
        #shutil.rmtree(dest)
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

def prepare_runfiles(self, dt:int) -> None:
    """Prepare files specific to this dt run"""
    # fill template files with per-run data:
    log.warn('REPEATING SPINUP FOR EACH DT !!!')
    #restart = '0' if dt == 0 else '1'
    restart='0'

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

    insfile = fill_template( os.path.join(self._dest, LPJGUESS_INS_FILE_TPL), run_data )
    open(os.path.join(self._dest, 'lpjguess.ins'), 'w').write(insfile)

class DynVeg_LpjGuess(Component):
    """classify a DEM in different landform, according to slope, elevation and aspect"""

    def __init__(self, dest:str):
        self._spinup = True
        self._timesteps = [0]
        self._dest = dest
        prepare_input(self._dest)

    @property
    def spinup(self):
        return self._spinup
    
    @property
    def timestep(self):
        '''Current timestep of sim'''
        if len(self._timesteps) > 0:
            return self._timesteps[-1]
        return None

    @property
    def elapsed(self):
        '''Total sim time elapsed'''
        return sum(self._timesteps)

    def run_one_step(self, dt:int=100) -> None:
        '''Run one lpj simulation step (duration: dt)'''
        self.prepare_runfiles(self.timestep // dt)
        self.generate_landform_files()
        self.execute_lpjguess()
        self.move_state()
        if self.timestep == 0:
            self._spinup = False
        self._timesteps.append( dt )

DynVeg_LpjGuess.prepare_runfiles = prepare_runfiles
DynVeg_LpjGuess.generate_landform_files = generate_landform_files
DynVeg_LpjGuess.execute_lpjguess = execute_lpjguess
DynVeg_LpjGuess.move_state = move_state


if __name__ == '__main__':
    # silence debug logging by setup loglevel to INFO here
    logging.getLogger().setLevel(logging.INFO)
    log.info('DynVeg LPJ-Guess Component')
    DT = 100
    component = DynVeg_LpjGuess(LPJGUESS_INPUT_PATH)

    for i in range(2):
        component.run_one_step(dt=DT)
