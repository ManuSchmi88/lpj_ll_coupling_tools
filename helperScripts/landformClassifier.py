"""
Created on Mon, March 5.

Author: Manuel Schmid
Manuel.Schmid@Uni-Tuebingen.de


Class uses a rectanglular grid DEM and calculates the topographic position index
(TPI) and a landform classification (after Weiss 2001). The functions to run the class
on a normal .csv-style input dem are theoretically there (see in the code marked as
deprecated) but the main usage is to load it as a landlab module.

Class has a landlab style main method "run_one_step(scalefactor)" which can be
used in the main loop to contiously write the tpi index and the landscape class
to the RasterModelGrid Instance in landlab, which is neat because then you can use
the standart NetCDF output to write data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import generic_filter
import math

class landformClassifier():
    """classify a DEM in different landform, according to slope, elevation and aspect"""

    def __init__(self, grid):
        """
        Constructor for the component.
        """
        self._grid = grid #loads the main dem
        self._dem = self._grid.at_node['topographic__elevation']
        self._slope = self._grid.at_node['topographic__steepest_slope']
        #convert slope map to degrees for CLASSIFICATION
        self._slope = np.rad2deg(self._slope)
        self._aspect = self._grid.calc_aspect_at_node()

        #set up landlab grid structure.
        self._grid.add_zeros('node', 'topographic_position__index')
        self._grid.add_zeros('node', 'topographic_position__class')
        self._grid.add_zeros('node', 'elevation__ID')
        self._grid.add_zeros('node', 'landform__ID')

        #Flags
        self._tpiTYPE = ''
        self.isPadded = False 
        #DEPRECATED
        #self.isPadded = False #flag for dem-padding
        #self.existGrid = False #flag if grid was initialised
        #self.existSlope = False #flag if slope-map was created
        NODATA = -9999


    """
    ---------------------------------------------------------------------------
    HERE ARE THE DEPRECATED FUNCTIONS WHICH WORK ON A NORMAL .CSV FORMATTED DEM
    FOR THE LANDLAB MODULE WE WILL USE THE LANDLAB NATIVE FUNCTIONS.
    AT SOME POINT JUST MOVE THIS TO A OWN NIFTY LITTLE CLASS
    ----------------------------------------------------------------------------
    """

    def _padGrid(self):
        """Enlarges grid by 2 nodes on each axis for proper slope calculation."""

        ny, nx = self._grid.shape #get the shape of the original grid
        _tempGrid = np.zeros((ny + 2, nx + 2)) #create enlarged grid
        _tempGrid[1:-1, 1:-1] = self._grid #insert old grid in center

        #values at the new padded boundaries will get the same values as their neigbours.
        #note: these added nodes don't have any physical meaning
        #sides
        _tempGrid[0 , 1:-1]  = self._grid[0 , :]
        _tempGrid[-1 , 1:-1] = self._grid[-1, :]
        _tempGrid[1:-1 , 0]  = self._grid[:, 0]
        _tempGrid[1:-1 , -1] = self._grid[:,-1]

        #corners
        _tempGrid[0,0]   = self._grid[0,0]
        _tempGrid[-1,-1] = self._grid[-1,-1]
        _tempGrid[0,-1]  = self._grid[0,-1]
        _tempGrid[-1,0]  = self._grid[-1,0]

        #copy the local _tempGrid to a global self.paddedGrid array
        self.paddedGrid = _tempGrid

        #set Flag if function was run, standart is False
        self.isPadded = True


    def _calcFiniteDifferences(self, dx):
        """Calculate the finite differences at each node for the 8-neighbour nodes.
        This is a implementation of the standart D8-Algorithm used in ArcGIS"""

        #if padGrid() was run then use the padded grid, otherwise use normal grid
        if self.isPadded == True:
            _dem = self.paddedGrid
        elif self.isPadded == False:
            _dem = self._grid


        (rows, cols) = np.shape(_dem)
        self.dzdx = np.zeros((rows,cols)) #create arrays for x-differences
        self.dzdy = np.zeros((rows,cols)) #create arrays for y-differences


        for i in range(1,rows-1):
            for j in range(1,cols-1):
                #calculate x-component
                dzdx = (((_dem[i-1,j+1]) + (2*_dem[i,j+1]) + (_dem[i+1,j+1])) -\
                    ((_dem[i-1,j-1]) + (2*_dem[i,j-1]) + (_dem[i+1,j-1]))) / (8*dx)
                #calculate y-component
                dzdy = (((_dem[i+1,j-1]) + (2*_dem[i+1,j]) + (_dem[i+1,j+1])) -\
                    ((_dem[i-1,j-1]) + (2*_dem[i-1,j]) + (_dem[i-1,j+1]))) / (8*dx)

                self.dzdx[i,j] = dzdx
                self.dzdy[i,j] = dzdy


    def _calcSlope(self, mode):
        """
        Takes the dzdx/dzdy arrays to calculate a slope in either rad or deg.
        mode = 'deg' :
            gives degree array
        mode = 'rad' :
            gives radians array
        """
        #if padGrid() was run then use the padded grid, otherwise use normal grid
        if self.isPadded == True:
            _dem = self.paddedGrid
        elif self.isPadded == False:
            _dem = self._grid

        (rows,cols) = np.shape(_dem)
        self.slope = np.zeros((rows,cols))

        while True:
            if mode == 'deg':
                self.slope = np.arctan(np.sqrt((self.dzdx**2) + (self.dzdy**2))) * (180/np.pi)
                break
            elif mode == 'rad':
                self.slope = np.arctan(np.sqrt((self.dzdx**2) + (self.dzdy**2)))
                break
            else:
                raise NameError('This is not a correct argument. Check function docstring')

        self.existSlope = True #set a flag to show that slope was calculated


    def _calcHillshade(self, azimuth, angle_altitude):
        """
        Calculates a hillshade map of a DEM

        Inputs are: azimuth and angle of sun relative to planar earth surface.
        """

        #check if grid was padded
        if self.isPadded == True:
            _dem = self.paddedGrid
        elif self.isPadded == False:
            _dem = self._grid

        azimuth = 360.0 - azimuth

        x,y = np.gradient(_dem)
        slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
        aspect = np.arctan(-x,y)
        azimuthrad = azimuth * np.pi/180.
        altituderad = angle_altitude * np.pi / 180

        shaded = np.sin(altituderad) * np.sin(slope)\
            + np.cos(altituderad) * np.cos(slope)\
            * np.cos((azimuthrad - np.pi / 2.) - aspect)

        self.hillshade = 255*(shaded + 1)/2.



    """
    ----------------------------------------------------------------------------
    HERE ARE FUNCTIONS WHICH ARE TUNED FOR USE IN LANDLAB
    ----------------------------------------------------------------------------
    """

    def reshapeGrid(self,nrows, ncols):

        """
        a little bit hacky right now. Landlab uses a 1D-array of elevation values
        to save data structure but we need a grid structure to run the landform
        classifier

        inputs:
            - grid : native landlab grid
            - nrows, ncols : number of rows and columns in grid
        """

        _gridRaw    = self._dem
        _slopeRaw   = self._slope
        _aspectRaw  = self._aspect

        _gridRe     = _gridRaw.reshape(nrows, ncols)
        _slopeRe    = _slopeRaw.reshape(nrows,ncols)
        _aspectRe   = _aspectRaw.reshape(nrows, ncols)
        self._dem   = _gridRe
        self._slope = _slopeRe
        self._aspect = _aspectRe

        return _gridRe, _slopeRe, _aspectRe

    def create_kernel(self,radius=2, invert=False):
        """Define a kernel
        After C.Werners lpjguesstools
        """

        if invert:
            value = 0
            k = np.ones((2*radius+1, 2*radius+1))
        else:
            value = 1
            k = np.zeros((2*radius+1, 2*radius+1))

        y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x**2 + y**2 <= radius**2
        k[mask] = value

        return k

    def _calcAspect(self, mode):
        """
        DEPRECATED. USE calcAspect() instead, it uses build in landlab routines


        Takes the self.dzdx and self.dzdy values and calculates a per-node aspect value of the grid.
        Algorithm is ESRI ArcGIS Standard:

            Aspect = 180/Pi * arctan2([dzdy])
        """

        aspect = (180/np.pi) * np.arctan2(self.dzdy, -self.dzdx)
        aspect = np.mod(450 - aspect, 360)
        aspect[aspect == 360] = 0

        self._aspect = aspect

    def calcAspect(self):
        """
        Uses build-in landlab candy to get aspect values of the model grid
        """

        aspect = self._grid.calc_aspect_at_node()
        #this was moved to the .reshapeGrid() function for consitency.
        #need to reshape here directly...
        #aspect = aspect.reshape(self._grid.number_of_node_rows,
        #             self._grid.number_of_node_columns)
        self._aspect = aspect.reshape(self._grid.number_of_node_rows, self._grid.number_of_node_columns)

    def classifyAspect(self, classNum = 4):
        #print('aspect classifier was run!')
        """
        Classifies aspect into aspect_classes

        Can use classType = 8:
                    1.N
                    2.NE
                    3.E
                    4.SE
                    5.S
                    6.SW
                    7.W
                    8.NW

                classType = 4:
                    1.N
                    2.E
                    3.S
                    4.W
                """
        while True:
            if classNum == '8':
                #define value breaks for aspect classes
                aspectClass = {1 : [0, 22.5],       #N
                               2 : [22.5, 67.5],    #NE
                               3 : [67.5,  112.6],  #E
                               4 : [112.5, 157.6],  #SE
                               5 : [157.5, 202.6],  #S
                               6 : [202.5, 247.6],  #SW
                               7 : [247.5, 292.6],  #W
                               8 : [292.5, 337.6],  #NW
                               9 : [337.5, 360]}    #N


                #create aspectClassArray
                #if padGrid() was run then use the padded grid, otherwise use normal grid
                if self.isPadded == True:
                    _dem = self.paddedGrid
                elif self.isPadded == False:
                    _dem = self._grid
                (rows,cols) = np.shape(_dem)
                self._aspectClass = np.zeros((rows,cols))

                #check which node belongs to which class
                for nrow in range(rows):
                    for ncol in range(cols):
                        for i,j in zip(aspectClass,aspectClass.values()):
                            if j[0] <= self._aspect[nrow,ncol] <= j[1]:
                                aspectC = i
                                if aspectC == 9:
                                    aspectC = 1
                                self._aspectClass[nrow,ncol] = aspectC
                break

            elif classNum == '4':
                #define value breaks for aspect classes
                aspectClass = {1 : [0,   45],       #N
                               2 : [45,  135],      #E
                               3 : [135, 225],      #S
                               4 : [225, 315],      #W
                               5 : [315, 360]}      #N

                #create aspectClassArray
                #if padGrid() was run then use the padded grid, otherwise use normal grid
                if self.isPadded == True:
                    _dem = self.paddedGrid
                elif self.isPadded == False:
                    _dem = self._grid
                (rows,cols) = np.shape(_dem)
                self._aspectClass = np.zeros((rows,cols))

                #check which node belongs to which class
                for nrow in range(rows):
                    for ncol in range(cols):
                        for i,j in zip(aspectClass,aspectClass.values()):
                            if j[0] <= self._aspect[nrow,ncol] <= j[1]:
                                aspectC = i
                                if aspectC == 5:
                                    aspectC = 1
                                self._aspectClass[nrow,ncol] = aspectC
                break

            else:
                raise NameError('This is not a correct argument. Check function docstring') 

            

    def calculate_tpi(self,scalefactor, res=30, return_unclassed=False, TYPE='SIMPLE'):
        """Classify DEM to tpi300 array according to Weiss 2001
        Modified after C. Werner, lpjguesstools.

        Be aware that this method is not per-se using landlab native methods
        but relies on import from landlab data and calculation by numpy.
        This means that the boundarynodes, which in landlab are marked as inactive
        and will not be considered by the landlab.imshow_grid method, are not inactive
        here.
        This mapping and conversion happens in the function ".writeTpiToGrid"
        """

        # Parameters:
        # - scalefactor: outerradius in map units (300 := 300m)
        # - res: resolution of one pixel in map units (default SRTM1: 30m)
        # - return_unclassed: return the continuous tpi values
        # - dx: cell size

        #data structures moved here for convenience
        NODATA = -9999

        # inner and outer tpi300 kernels
        k_smooth = self.create_kernel(radius=2)

        radius_outer = int(math.ceil(scalefactor / float(res)))
        radius_inner = int(math.floor((scalefactor - 150)/float(res)))

        k_outer = self.create_kernel(radius=radius_outer)
        #k_inner = self.create_kernel(radius=radius_inner, invert=True)
        if radius_inner > 0:
            k_inner = self.create_kernel(radius=radius_inner, invert=True)
        else:
            raise ValueError("Negative radius of inner kernel. Adjust Scalefactor.")

        x = y = int((k_outer.shape[0] - k_inner.shape[0]) / 2)
        k_outer[x:x+k_inner.shape[0], y:y+k_inner.shape[1]] = k_inner


        # compute tpi
        tpi = self._dem - generic_filter(self._dem, np.mean, footprint=k_outer, mode="reflect") + 0.5
        tpi = generic_filter(tpi, np.mean, footprint=k_smooth, mode="reflect").astype(int)

        if TYPE == 'WEISS':
            self._tpiTYPE = "WEISS"
            # values from poster, results in strange NaN values for some reason beyound me.
            mz10, mz05, pz05, pz10 = np.percentile(tpi, [100-84.13, 100-69.15, 69.15, 84.13])
            #TODO: check if this should be a decision tree (we have unclassified cells)
            tpi_classes = np.ones( tpi.shape ) * NODATA
            tpi_classes[(tpi > pz10)]                                   = 1 # ridge
            tpi_classes[((tpi > pz05)  & (tpi <= pz10))]                = 2 # upper slope
            tpi_classes[((tpi > mz05)  & (tpi <  pz05) & (self._slope >  5))] = 3 # middle slope
            tpi_classes[((tpi >= mz05) & (tpi <= pz05) & (self._slope <= 5))] = 4 # flats slope
            tpi_classes[((tpi >= mz10) & (tpi <  mz05))]                = 5 # lower slopes
            tpi_classes[(tpi < mz10)]                                   = 6 # valleys

        # simplified:
        if TYPE == 'SIMPLE':
            self._tpiTYPE = 'SIMPLE'
            # according to Tagil & Jenness (2008) Science Alert doi:10.3923/jas.2008.910.921
            mz10, pz10 = np.percentile(tpi, [100-84.13, 84.13])
            tpi_classes = np.ones( tpi.shape ) * NODATA
            tpi_classes[(tpi >= mz10)]                               = 1 # hilltop
            tpi_classes[(tpi >= mz10) & (tpi < pz10) & (self._slope >= 6)] = 3 # mid slope
            tpi_classes[(tpi >  mz10) & (tpi < pz10) & (self._slope  < 6)]  = 4 # flat surface
            tpi_classes[(tpi <= mz10)]                               = 6 # valley

        if return_unclassed:
            return tpi

        self._tpiClasses = tpi_classes
        self._tpi = tpi
        #return tpi_classes

    aspectLF = [2,3,5]
            

    def writeTpiToGrid(self):
        """
        Function that takes tpi-values and tpi classes and writes it back to
        Landlab Grid nodes
        """

        #flatten array to make it landlab accesible
        _tpiFlat = self._tpi.flatten().astype(int)
        _tpiClassFlat = self._tpiClasses.flatten().astype(int)

        #write data to array
        for struct,data in zip(['topographic_position__index', 'topographic_position__class'],
                        [_tpiFlat, _tpiClassFlat]):
            self._grid.at_node[struct] = data
            self._grid.at_node[struct][self._grid.boundary_nodes] = 0

    def updateGrid(self):
        """
        Call this to reload Grid-parameters like slope and elevation.
        This takes time. Is there a more efficient way to do this?
        Is this even needed? Is this the real life or is this just fantasy.
        caught in a bugslide, no escape from morphology.
        """

        self._dem = self._grid.at_node['topographic__elevation']
        self._slope = np.rad2deg(self._grid.at_node['topographic__steepest_slope'])
        self._aspect = self._grid.calc_aspect_at_node()
        
        
    def createElevationID(self, dem, minimum,maximum,step):
        elevationID = np.zeros(np.shape(dem)) #creates ID array
        elevationSteps = np.arange(minimum,maximum,step) #creates elevation-step array
        counterID = 1 #starts at 1 for lowester elevation class
    
        for i in elevationSteps:
            index = np.where((dem >= i) & (dem < i+step))
            elevationID[index] = counterID
            counterID += 1
            

        self._elevationID = elevationID
        self._grid.at_node['elevation__ID'] = elevationID   

        return elevationID

    def createLandformID(self):
        """
        creates a per-node grid with a 3-integer landform-ID.

        ID-system is:
            [elevationID, slopeID, aspectID]
            
            elevationID:
                1 = [minium:step[i]] e.g   1 = [0:200]
                                            2 = [200:400]
                                            3 = [400 : 600]
            
            slopeID: TPI Slope ID, see TPI function
            
            aspectID:
            right now it only works with the 4-class aspect ID system for consistency with LPJGuess
                1 N
                2 E
                3 S
                4 W

        """

        #this is not very memory efficient and was just done for ease of coding
        #can be changed later but due to pythons fancy indexing I guess this is still
        #ok. 
        _slopeID     = self._tpiClasses.flatten().astype(int)
        _aspectID    = self._aspectClass.flatten().astype(int)
        _elevationID = self._elevationID.flatten().astype(int)

        #check which tpi_type was used and adjust the matrix with aspect-dependend landform

        if self._tpiTYPE == 'WEISS':
            lfClasses = [2,3,5]
        elif self._tpiTYPE == 'SIMPLE':
            lfClasses = [3]


        #this whole int(str(int())) thing is probably horrible but thats what it is for now...
        #works though...
        for i in range(len(self._grid.at_node['landform__ID'])):
            if _slopeID[i] in lfClasses:
                lfIndex = int(str(_elevationID[i]) + str(_slopeID[i]) + str(_aspectID[i]))
            else:
                lfIndex = int(str(_elevationID[i]) + str(_slopeID[i]) + str(0))

            self._grid.at_node['landform__ID'][i] = lfIndex


    def run_one_step(self, scalefact, classtype):
            """
            Landlab style wrapper function which is to be called in the main-model-loop
            """
            #self.createLandlabDatafields() #moved to constructor.
            self.updateGrid()
            self.reshapeGrid(nrows = self._grid.number_of_node_rows,
                            ncols = self._grid.number_of_node_columns)
            self.calculate_tpi(scalefact, res = self._grid.dx, TYPE = classtype)
            self.calcAspect()
            self.classifyAspect(classNum = '4')
            self.createElevationID(self._dem, 0, 6000, 200) #hardcoding of value... BAD!
            self.createLandformID()
            self.writeTpiToGrid()

        