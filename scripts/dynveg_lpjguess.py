from enum import Enum
from landlab import Component
import logging
import xarray as xr
import sys

logPath = '.'
fileName = 'dynveg_lpjguess.log'

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

def split_climate(ds_files, dt:int, time_step:TS=TS.MONTHLY):
    """Split climte files into dt-length chunks"""
    for ds_file in ds_files:
        ds = xr.open_dataset(ds_file, decode_times=False)

        # do something


class DynVeg_LpjGuess(Component):
    """classify a DEM in different landform, according to slope, elevation and aspect"""

    @property
    def spinup(self):
        return self._spinup
    
    @property
    def timestep(self):
        return self._current_timestep

    def __init__(self, spinup:bool = False):
        self._spinup = spinup
        self._current_timestep = 0


    def run_one_step(self) -> None:
        pass



def test_dynveg_contructor():
    c = DynVeg_LpjGuess()
    print(c.spinup)
    print(c)

if __name__ == '__main__':
    log.info('Starting dynveg lpjguess component')
    test_dynveg_contructor()