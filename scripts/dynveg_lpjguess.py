from enum import Enum
from landlab import Component
import xarray as xr

channel = logging.StreamHandler(sys.stdout)
channel.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
channel.setFormatter(formatter)
log.addHandler(ch)


class TS(Enum):
    DAILY = 1
    MONTHLY = 2

def split_climate(ds_files, dt:int, time_step:TS=TS.MONTHLY):
    """Split climte files into dt-length chunks"""
    for ds_file in ds_files:
        ds = xr.open_dataset(ds_file, decode_times=False)


class DynVeg_LpjGuessSetup():

    @classmethod
    def create_folders(self):
        pass

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
        self._current_timestep


    def run_one_step(self) -> None:
        pass



def test_dynveg_contructor():
    c = DynVeg_LpjGuess()
    print(c.spinup)
    print(c)

if __name__ == '__main__':
    test_dynveg_contructor()