"""Landlab Driver for running Landscape Evolution Experiments with
    - Soil weathering
    - Soil diffusion
    - Detachment-limited river erosion
    - tectonic uplift
    - vegetation modulation of erosion effects

Created by: Manuel Schmid, University of Tuebingen, 07.04.2017
"""

## Import necessary Python and Landlab Modules
import numpy as np
from landlab import RasterModelGrid
from landlab import CLOSED_BOUNDARY, FIXED_VALUE_BOUNDARY
from landlab.components.flow_routing import FlowRouter
from landlab.components import ExponentialWeatherer
from landlab.components import DepthDependentDiffuser
from landlab.components import LinearDiffuser
from landlab.components import FastscapeEroder
from landlab.components import Space
from landlab.components import DepressionFinderAndRouter
from landlab.components import drainage_density
from landlab.components import SteepnessFinder
from landlab.components import rainfallOscillation as ro
from landlab import imshow_grid
from landlab.components import landformClassifier
from landlab.io.netcdf import write_netcdf
from landlab.io.netcdf import read_netcdf
from create_input_for_landlab import lpj_import_run_one_step 
from dynveg_lpjguess import *
from matplotlib import pyplot as plt
from matplotlib import rcParams
import logging, time
from timer import timed
rcParams.update({'figure.autolayout': True})
rcParams['agg.path.chunksize'] = 200000000
import time
#import the .py-inputfile
from inputFile import *


#define consts - source environemt.sh
#LPJGUESS_INPUT_PATH = os.environ.get('LPJGUESS_INPUT_PATH', './temp_lpj')
#LPJGUESS_TEMPLATE_PATH = os.environ.get('LPJGUESS_TEMPLATE_PATH', './lpjguess.template')
#LPJGUESS_FORCINGS_PATH = os.environ.get('LPJGUESS_FORCINGS_PATH', './forcings')
#LPJGUESS_INS_FILE_TPL = os.environ.get('LPJGUESS_INS_FILE_TPL', 'lpjguess.ins.tpl')
#LPJGUESS_BIN = os.environ.get('LPJGUESS_BIN', '/esd/esd01/data/mschmid/coupling/build/guess')
#LPJGUESS_CO2FILE = os.environ.get('LPJGUESS_CO2FILE', 'co2_TraCE_egu2018_35ka_const180ppm.txt')
##DIRTY
LPJGUESS_INPUT_PATH = './temp_lpj'
LPJGUESS_TEMPLATE_PATH = './lpjguess.template'
LPJGUESS_FORCINGS_PATH = './forcings'
LPJGUESS_INS_FILE_TPL = 'lpjguess.ins.tpl'
LPJGUESS_BIN = '/esd/esd01/data/mschmid/coupling/build/guess'
LPJGUESS_CO2FILE = 'co2_TraCE_egu2018_35ka_const180ppm.txt'

#timelogging for debugs
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


#input-processing:
#Number of total-timestep (nt) and spin-up timesteps (ssnt)
nt = int(totalT / dt)
#time-vector (total and transient), used for plotting later
timeVec = np.arange(0, totalT, dt)
transTimeVec = np.arange(0, (totalT - ssT), dt)
transientRainfallTimespan = int(totalT - ssT)
#calculate the uplift per timestep
uplift_per_step = upliftRate * dt
#Number of total produced outputs
no = totalT / outInt
#number of zeros for file_naming. Don't meddle with this.
zp = len(str(int(no)))

print("finished with parameter-initiation")
print("---------------------")


#---------------------------------Grid Setup-----------------------------------#
#This initiates a Modelgrid with dimensions nrows x ncols and spatial scaling of dx
mg = RasterModelGrid((nrows,ncols), dx)

#only uncomment this if there is a pre-existing topography you want to load. 
#right now this only works if the topo was saved in numpys .npy format.
try:
    topoSeed = np.load('topoSeed.npy')
    print('loaded topoSeed.npy')
except:
    print('There is no file containing a initial topography')

#Initate all the fields that are needed for calculations
mg.add_zeros('node', 'topographic__elevation')
mg.add_zeros('node', 'bedrock__elevation')
mg.add_zeros('node', 'soil_production__rate')
mg.add_zeros('node', 'soil__depth')
mg.add_zeros('node', 'tpi__mask')
mg.add_zeros('node', 'erosion__rate')
mg.add_zeros('node', 'median_soil__depth')

#checks if standart topo is used. if not creates own
if 'topoSeed' in locals():
    topo_tilt = mg.node_y/100000000 + mg.node_x/100000000
    mg.at_node['topographic__elevation'] += (topoSeed + initialSoilDepth +
            topo_tilt) 
    mg.at_node['bedrock__elevation'] += (topoSeed + topo_tilt)
    mg.at_node['soil__depth'] += initialSoilDepth
    print('Using pre-existing topography from file topoSeed.npy')

else:
    topo_tilt = mg.node_y/100000000 + mg.node_x/100000000
    mg.at_node['topographic__elevation'] += (np.random.rand(mg.at_node.size)/10000 + initialSoilDepth)
    mg.at_node['topographic__elevation'] += topo_tilt
    mg.at_node['bedrock__elevation'] += (np.random.rand(mg.at_node.size)/10000 + initialSoilDepth)
    mg.at_node['bedrock__elevation'] += topo_tilt
    mg.at_node['soil__depth'] += initialSoilDepth
    print('No pre-existing topography. Creating own random noise topo.')

print('Creating soil layer under bedrock layer with {}m thickness'.format(initialSoilDepth))

mg.add_zeros('node','vegetation__density')

#Create boundary conditions of the model grid (eeither closed or fixed-head)
for edge in (mg.nodes_at_left_edge,mg.nodes_at_right_edge,
        mg.nodes_at_top_edge, mg.nodes_at_bottom_edge):
    mg.status_at_node[edge] = CLOSED_BOUNDARY

#Create one single outlet node
mg.set_watershed_boundary_condition_outlet_id(0,mg['node']['topographic__elevation'],-9999)
#create mask datafield which defaults to 1 to all core nodes and to 0 for
#booundary nodes. LPJGUESS needs this
mg.at_node['tpi__mask'][mg.core_nodes] = 1
mg.at_node['tpi__mask'][mg.boundary_nodes] = 0

print("finished with setup of modelgrid")
print("---------------------")

##---------------------------------Vegi implementation--------------------------#
##Set up a timeseries for vegetation-densities
vegiTimeseries  = np.zeros(int(totalT / dt)) + vp
#this incorporates a vegi step-function at timestep sfT with amplitude sfA
mg.at_node['vegetation__density'][:] = vp
#This maps the vegetation density on the nodes to the links between the nodes
vegiLinks = mg.map_mean_of_link_nodes_to_link('vegetation__density')

##These are the necesseray calculations for implementing the vegetation__density
##in the fluvial routines
nSoil_to_15 = np.power(nSoil, 1.5)
Ford = aqDens * grav * nSoil_to_15
n_v_frac = nSoil + (nVRef * ((mg.at_node['vegetation__density'] / vRef)**w)) #self.vd = VARIABLE!
Prefect = np.power(n_v_frac, 0.9)
Kvs = k_sediment * Ford/Prefect
Kvb = k_bedrock  * Ford/Prefect

##These are the calcultions to calculate the linear diffusivity based on vegis
linDiff = mg.zeros('node', dtype = float)
linDiff = linDiffBase * np.exp(-alphaDiff * vegiLinks)

print("finished setting up the vegetation fields and Kdiff and Kriv")
print("---------------------")

##---------------------------------Rain implementation--------------------------#
##Set up a Timeseries of rainfall values
rainTimeseries = np.zeros(int(totalT / dt)) + baseRainfall
mg.add_zeros('node', 'rainvalue')
mg.at_node['rainvalue'][:] = int(baseRainfall)


##---------------------------------Array initialization---------------------#
##This initializes all the arrays that are used to store data during the runtime
##of the model. this is mostly for plotting purposed and to create the .txt
##outputs. This potentially takes up a lot of space, so check if needed.
dhdtA       = [] #Vector containing dhdt values for each node per timestep
meandhdt    = [] #contains mean elevation change per timestep
mean_E      = [] #contains the mean "erosion" rate out of Massbalance
mean_hill_E = [] #contains mean hillslope erosion rate
mean_riv_E  = [] #contains mean river erosion rate
mean_dd     = [] #contains mean drainage density
mean_K_riv  = [] #contains mean K-value for spl
mean_K_diff = [] #contains mean K-value for ld
mean_slope  = [] #mean slope within model-area
max_slope   = [] #maximum slope within model area
min_slope   = [] #minimum slope within model area
mean_elev   = [] #mean elevation within model area
max_elev    = [] #maximum elevation within model area
min_elev    = [] #minimum elevation within model area
vegi_P_mean = [] #mostly for bugfixing because Manu is stupid fuckup without brain and life and fuck you
mean_SD     = [] #mean soil depth
mean_Ksn    = [] #mean channel steepness
max_Ksn     = [] #max channel steepness

##---------------------------------Component initialization---------------------#


fr = FlowRouter(mg,method = 'd8', runoff_rate = baseRainfall)

lm = DepressionFinderAndRouter(mg)

expWeath = ExponentialWeatherer(mg, soil_production__maximum_rate =
        soilProductionRate, soil_production__decay_depth = 2.5)

sf = SteepnessFinder(mg,
                    min_drainage_area = 1e6)

sp = Space(mg, K_sed=Kvs, K_br=Kvb, 
           F_f=Ff, phi=phi, H_star=Hstar, v_s=vs, m_sp=m, n_sp=n,
           sp_crit_sed=sp_crit_sedi, sp_crit_br=sp_crit_bedrock,
           solver = solver)

lc = landformClassifier(mg)

DDdiff = DepthDependentDiffuser(mg, 
            linear_diffusivity = linDiff,
            soil_transport_decay_depth = 2)

#print('lpjinputpath :    ' + LPJGUESS_INPUT_PATH)
print('lpjco2file :' + LPJGUESS_CO2FILE)
lpj = DynVeg_LpjGuess(LPJGUESS_INPUT_PATH)

print("finished with the initialization of the erosion components")   
print("---------------------")

##---------------------------------Main Loop------------------------------------#
t0 = time.time()
elapsed_time = 0
print("starting with main loop.")
print("---------------------")
#Create incremental counter for controlling progress of mainloop
counter = 0
#Create Limits for DHDT plot. Move this somewhere else later..
DHDTLowLim = upliftRate - (upliftRate * 1)
DHDTHighLim = upliftRate + (upliftRate * 1)

while elapsed_time < totalT:

    #create copy of "old" topography
    z0 = mg.at_node['topographic__elevation'].copy()

    #Call the erosion routines.
    fr.run_one_step()
    lm.map_depressions()
    floodedNodes = np.where(lm.flood_status==3)[0]
    sp.run_one_step(dt = dt, flooded_nodes = floodedNodes)
    expWeath.calc_soil_prod_rate()
    DDdiff.run_one_step(dt=dt)
    lc.run_one_step(elevationStepBin, 300, classtype = classificationType)
    lpj_import_run_one_step(mg, '../input/sp_lai.out', method = 'cumulative')
    lpj.run_one_step(dt=dt)

    #apply uplift
    mg.at_node['bedrock__elevation'][mg.core_nodes] += uplift_per_step
    #mg.at_node['topographic__elevation'][mg.core_nodes] += uplift_per_step
    
    #set soil-depth to zero at outlet node
    #TODO: try disabling this to get non-zero soil depth?
    mg.at_node['soil__depth'][0] = 0
    
    #add newly weathered soil
    #mg.at_node['soil__depth'][:] += \
    #        (mg.at_node['soil_production__rate'][:] * dt)

    #recalculate topographic elevation
    mg.at_node['topographic__elevation'][:] = \
            mg.at_node['bedrock__elevation'][:] + mg.at_node['soil__depth'][:]

    #Calculate median soil-depth
    #for ids in np.unique(mg.at_node['landform__ID'][:]):
    #    _soilIDS = np.where(mg.at_node['landform__ID']==ids)
    #    mg.at_node['median_soil__depth'][_soilIDS] = np.median(mg.at_node['soil__depth'][_soilIDS]) 

    #calculate drainage_density
    channel_mask = mg.at_node['drainage_area'] > critArea
    dd = drainage_density.DrainageDensity(mg, channel__mask = channel_mask)
    mean_dd.append(dd.calc_drainage_density())

    #Calculate dhdt and E
    dh = (mg.at_node['topographic__elevation'] - z0)
    dhdt = dh/dt
    erosionMatrix = upliftRate - dhdt
    mg.at_node['erosion__rate'] = erosionMatrix
    mean_E.append(np.mean(erosionMatrix))

    #Calculate river erosion rate, based on critical area threshold
    dh_riv = mg.at_node['topographic__elevation'][np.where(mg.at_node['drainage_area'] > critArea)]\
        - z0[np.where(mg.at_node['drainage_area'] > critArea)]
    dhdt_riv = dh_riv/dt
    mean_riv_E.append(np.mean(upliftRate - dhdt_riv))

    #Calculate hillslope erosion rate
    dh_hill = mg.at_node['topographic__elevation'][np.where(mg.at_node['drainage_area'] <= critArea)]\
        - z0[np.where(mg.at_node['drainage_area'] <= critArea)]
    dhdt_hill = dh_hill/dt
    mean_hill_E.append(np.mean(upliftRate - dhdt_hill))

    vegiLinks = mg.map_mean_of_link_nodes_to_link('vegetation__density')
    #update LinearDiffuser
    linDiff = linDiffBase*np.exp(-alphaDiff * vegiLinks)
    #reinitalize Diffuser
    ld   = LinearDiffuser(mg, linear_diffusivity = linDiff) 

    #update K_sp
    n_v_frac = nSoil + (nVRef * (mg.at_node['vegetation__density'] / vRef)) #self.vd = VARIABLE!
    n_v_frac_to_w = np.power(n_v_frac, w)
    Prefect = np.power(n_v_frac_to_w, 0.9)
    Kvs = k_sediment * Ford/Prefect
    Kvb = k_bedrock  * Ford/Prefect
    sp.K_sed = Kvs
    sp.K_bed = Kvb

    #Calculate and save mean, max, min slopes
    mean_slope.append(np.mean(mg.at_node['topographic__steepest_slope'][mg.core_nodes]))
    max_slope.append(np.max(mg.at_node['topographic__steepest_slope'][mg.core_nodes]))
    min_slope.append(np.min(mg.at_node['topographic__steepest_slope'][mg.core_nodes]))

    #calculate and save mean, max, min elevation
    mean_elev.append(np.mean(mg.at_node['topographic__elevation'][mg.core_nodes]))
    max_elev.append(np.max(mg.at_node['topographic__elevation'][mg.core_nodes]))
    min_elev.append(np.min(mg.at_node['topographic__elevation'][mg.core_nodes]))

    #Mean Ksn Value
    #_ksndump = mg.at_node['channel__steepness_index'][mg.core_nodes]
    #mean_Ksn.append(np.mean(_ksndump[np.nonzero(_ksndump)]))
    #max_Ksn.append(np.max(_ksndump[np.nonzero(_ksndump)]))

    counter += 1
    #print(counter)

    #Run the output loop every outInt-times
    if elapsed_time % outInt  == 0:

        print('Elapsed Time:' , elapsed_time,', writing output!')
        ##Create DEM
        plt.figure()
        imshow_grid(mg,'topographic__elevation',grid_units=['m','m'],var_name = 'Elevation',cmap='terrain')
        plt.savefig('./ll_output/DEM/DEM_'+str(int(elapsed_time/outInt)).zfill(zp)+'.png')
        plt.close()
        ##Create Bedrock Elevation Map
        plt.figure()
        imshow_grid(mg,'bedrock__elevation', grid_units=['m','m'], var_name = 'bedrock', cmap='jet')
        plt.savefig('./ll_output/BED/BED_'+str(int(elapsed_time/outInt)).zfill(zp)+'.png')
        plt.close()
        ##Create Slope - Area Map
        plt.figure()
        plt.loglog(mg.at_node['drainage_area'][np.where(mg.at_node['drainage_area'] > 0)],
           mg.at_node['topographic__steepest_slope'][np.where(mg.at_node['drainage_area'] > 0)],
           marker='.',linestyle='None')
        plt.xlabel('Area')
        plt.ylabel('Slope')
        plt.savefig('./ll_output/SA/SA_'+str(int(elapsed_time/outInt)).zfill(zp)+'.png')
        plt.close()
        ##Create NetCDF Output
        write_netcdf('./ll_output/NC/output{}'.format(elapsed_time)+'__'+str(int(elapsed_time/outInt)).zfill(zp)+'.nc',
                mg,format='NETCDF4', attrs = {'lgt.lat' : latitude,
                                              'lgt.lon' : longitude,
                                              'lgt.dx'  : dx,
                                              'lgt.dy'  : dx,
                                              'lgt.timestep' : elapsed_time,
                                              'lgt.classification' : classificationType,
                                              'lgt.elevation_step' : elevationStepBin})
        ##Create NetCDF Output
        write_netcdf('./temp_output/current_output.nc',
                mg,format='NETCDF4', attrs = {'lgt.lat' : latitude,
                                              'lgt.lon' : longitude,
                                              'lgt.dx'  : dx,
                                              'lgt.dy'  : dx,
                                              'lgt.timestep' : elapsed_time,
                                              'lgt.classification' : classificationType,
                                              'lgt.elevation_step' : elevationStepBin})
        ##Create erosion_diffmaps
        plt.figure()
        imshow_grid(mg,erosionMatrix,grid_units=['m','m'],var_name='Erosion m/yr',cmap='jet',limits=[DHDTLowLim,DHDTHighLim])
        plt.savefig('./ll_output/DHDT/eMap_'+str(int(elapsed_time/outInt)).zfill(zp)+'.png')
        plt.close()
        ##Create ACC Maps
        #plt.figure()
        #imshow_grid(mg, 'drainage__area', grid_units=['m','m'],
        #        var_name='ksn', cmap='jet')
        #plt.savefig('./Ksn/ksnMap_'+str(int(elapsed_time/outInt)).zfill(zp)+'.png')
        #plt.close()
        plt.figure()
        imshow_grid(mg,'soil__depth',grid_units=['m','m'],var_name=
                'Elevation',cmap='terrain')
        plt.savefig('./SoilDepth/SD_'+str(int(elapsed_time/outInt)).zfill(zp)+'.png')
        plt.close()
        #Create SoilProd Maps
        plt.figure()
        imshow_grid(mg,'soil_production__rate')
        plt.savefig('./SoilP/SoilP_'+str(int(elapsed_time/outInt)).zfill(zp)+'.png')
        plt.close()

    elapsed_time += dt #update elapsed time
tE = time.time()
print()
print('End of  Main Loop. So far it took {}s to get here. No worries homeboy...'.format(tE-t0))


##---------------------------------Plotting-------------------------------------#


#Plot Vegi_erosion_rate
fig, axarr = plt.subplots(6, sharex = True, figsize = [11,14])
axarr[0].plot(timeVec, vegiTimeseries,'g', linewidth = 2.5)
axarr[0].set_title('Mean Surface Vegetation', fontsize = 12)
axarr[0].set_ylabel('Vegetation cover')
axarr[1].plot(timeVec, mean_elev, 'k', linewidth = 2.5)
axarr[1].plot(timeVec, max_elev, 'k--', linewidth = 2, alpha = 0.5)
axarr[1].plot(timeVec, min_elev, 'k--', linewidth = 2, alpha = 0.5)
axarr[1].set_title('Mean Elevation', fontsize = 12)
axarr[1].set_ylabel('Mean Elevation [m]')
#axarr[1].set_ylim([0,80])
axarr[2].plot(timeVec, np.degrees(np.arctan(mean_slope)), 'r', linewidth = 2.5)
axarr[2].plot(timeVec, np.degrees(np.arctan(max_slope)), 'r--', linewidth = 2.0, alpha = 0.5)
axarr[2].plot(timeVec, np.degrees(np.arctan(min_slope)), 'r--', linewidth = 2.0, alpha = 0.5)
#axarr[2].set_ylim([0,10])
axarr[2].set_title('Mean Slope', fontsize = 12)
axarr[2].set_ylabel('Mean Slope [deg]')
axarr[3].plot(timeVec,mean_dd, 'b', linewidth = 2.5)
axarr[3].set_title('Mean Drainage Density')
axarr[3].set_ylabel('Drainage Density')
axarr[4].plot(timeVec, mean_hill_E, 'g--', linewidth = 2.0, alpha = 0.5)
axarr[4].plot(timeVec, mean_riv_E, 'b--', linewidth = 2.0, alpha = 0.5)
axarr[4].plot(timeVec, mean_E, 'r--', linewidth = 2.2, alpha = 0.8)
axarr[4].legend(['Hillsl.', 'Rivers','Mean'])
axarr[4].set_title("Erosion rates")
axarr[4].set_ylabel('Erosion rate [m/yr]')
axarr[5].plot(timeVec, rainTimeseries, 'k', linewidth = 2.5)
axarr[5].set_title("Rainfall")
axarr[5].set_ylabel("Rain Value")
axarr[5].set_xlabel("Model Years", fontsize = 12)
plt.savefig('./Multiplot_absolut.png',dpi = 720)
plt.close()

#Save the most useful output arrays as CSV file for later plotting
np.savetxt('./CSVOutput/MeanSlope.csv', mean_slope)
np.savetxt('./CSVOutput/MaxSlope.csv', max_slope)
np.savetxt('./CSVOutput/MeanElev.csv', mean_elev)
np.savetxt('./CSVOutput/MaxElev.csv', max_elev)
np.savetxt('./CSVOutput/MeanRiverErosion.csv', mean_riv_E)
np.savetxt('./CSVOutput/MeanHillslopeErosion.csv', mean_hill_E)
np.savetxt('./CSVOutput/MeanErosion.csv', mean_E)
np.savetxt('./CSVOutput/VegetationDensity.csv', vegiTimeseries)
np.savetxt('./CSVOutput/Timeseries.csv', timeVec)
np.savetxt('./CSVOutput/Vegi_bugfix.csv', vegi_P_mean)
np.savetxt('./CSVOutput/MeanSoilthick.csv', mean_SD)
np.savetxt('./CSVOutput/MeanRivK.csv', mean_K_riv)
np.savetxt('./CSVOutput/MeanHillK.csv', mean_K_diff)
#np.savetxt('./CSVOutput/MeanKsn.csv', mean_Ksn)
#np.savetxt('./CSVOutput/MaxKsn.csv', max_Ksn)
np.savetxt('./CSVOutput/RainTimeseries.csv', rainTimeseries)

#bugfixing because manu is a fucking dumb asshole
plt.plot(vegi_P_mean)
plt.savefig('./vegi_P_bugfix.png', dpi = 720)
plt.close()
print("FINALLY! TADA! IT IS DONE! LOOK AT ALL THE OUTPUT I MADE!!!!")
