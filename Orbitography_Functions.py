import orekit
vm = orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir, absolutedate_to_datetime
setup_orekit_curdir()

from org.orekit.data import DataProvidersManager, ZipJarCrawler # type: ignore
from org.orekit.frames import FramesFactory, TopocentricFrame # type: ignore
from org.orekit.bodies import OneAxisEllipsoid, GeodeticPoint # type: ignore
from org.orekit.time import TimeScalesFactory, AbsoluteDate, DateTimeComponents # type: ignore
from org.orekit.utils import IERSConventions, Constants # type: ignore
from org.orekit.propagation.analytical.tle import TLE, TLEPropagator # type: ignore
from org.orekit.propagation.numerical import NumericalPropagator # type: ignore
from org.orekit.propagation import SpacecraftState  # type: ignore
from org.orekit.orbits import OrbitType, PositionAngleType, CartesianOrbit # type: ignore
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator # type: ignore
from org.orekit.forces.gravity.potential import GravityFieldFactory # type: ignore
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel # type: ignore
from orekit import JArray_double

from java.util import Arrays # type: ignore
from java.io import File # type: ignore

import numpy as np
from datetime import datetime, timedelta, UTC
import plotly.graph_objs as go
from PIL import Image

class OrbitographyFunctions:

    def __init__(self, sat_data):
        self.sat_data   = sat_data
        self.eme2000    = FramesFactory.getEME2000()
        self.itrf       = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        now             = datetime.now(UTC)
        self.time_now   = AbsoluteDate(now.year, now.month, now.day, now.hour, now.minute + 1, 0.0, TimeScalesFactory.getUTC())

    def createSpheroid(self, texture, equatorial_radius, flattening=0):
        """
        Fetches the current time, rounded down to the minute

        Args:
            texture: 

            equatorial_radius: float
                The radius of the spheroid at the equator, in metres
            
            flattening: float, optional
                The flattening value of the spheroid. Leave empty to define a perfect sphere

        Returns:
            Array of radii in the x, y, and z directions, where z is up. Units are in metres
        """

        N_lat = int(texture.shape[0])
        N_lon = int(texture.shape[1])
        theta = np.linspace(0,2*np.pi,N_lat)
        phi   = np.linspace(0,np.pi,N_lon) 

        axes_radii = [equatorial_radius * np.outer(np.cos(theta),np.sin(phi)),
                    equatorial_radius * np.outer(np.sin(theta),np.sin(phi)),
                    equatorial_radius * (1 - flattening) * np.outer(np.ones(N_lat),np.cos(phi))
                    ]
        return axes_radii

    def plotEarth(self):
        """
        Creates a spherical image of the Earth
        """

        # Data required for rendering the Earth's texture, and rotates the planet to align prime meridian
        earth_texture = np.asarray(Image.open('earth.jpeg')).T
        earth_texture = np.concatenate((earth_texture[int(earth_texture.shape[0]/2):], earth_texture[:int(earth_texture.shape[0]/2)]))
        colorscale =[[0.0, 'rgb(30, 59, 117)'],
                     [0.1, 'rgb(46, 68, 21)'],
                     [0.2, 'rgb(74, 96, 28)'],
                     [0.3, 'rgb(115,141,90)'],
                     [0.4, 'rgb(122, 126, 75)'],
                     [0.6, 'rgb(122, 126, 75)'],
                     [0.7, 'rgb(141,115,96)'],
                     [0.8, 'rgb(223, 197, 170)'],
                     [0.9, 'rgb(237,214,183)'],
                     [1.0, 'rgb(255, 255, 255)']]

        axes = self.createSpheroid(earth_texture, Constants.IERS2010_EARTH_EQUATORIAL_RADIUS, Constants.IERS2010_EARTH_FLATTENING)
        surf = go.Surface(x=axes[0], y=axes[1], z=axes[2],
                        surfacecolor = earth_texture,
                        colorscale   = colorscale,
                        showscale    = False,
                        hoverinfo    = 'skip',
                        )    
        return surf

    def plotBackground(self):
        """
        Creates a black spherical background
        """

        bg_texture = np.asarray(Image.open('earth.jpeg')).T
        axes = self.createSpheroid(bg_texture, 100*Constants.IERS2010_EARTH_EQUATORIAL_RADIUS)
        surf = go.Surface(x=axes[0], y=axes[1], z=axes[2],
                        colorscale = [[0.0, 'black'],
                                      [1.0, 'black']],
                        showscale  = False,
                        hoverinfo  = 'skip',
                        )    
        return surf

    def getEpochDelta(self, sat):
        """
        Calculates time between the epoch of the current satellite and time_now

        Args:
            sat: the ID of the desired satellite
        Returns:
            TimeDelta of time since epoch
        """

        epoch = self.sat_data.get(sat)['TLE'].getDate()
        return timedelta(seconds = self.time_now.durationFrom(epoch))

    def earth(self, frame):
        return OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, 
                                Constants.WGS84_EARTH_FLATTENING, 
                                frame)

    def initialOrbitTLE(self, sat_tle):
        """
        Initialises a Cartesian orbit of an object, using TLE data

        Args:
            sat_tle: 
                TLE data of desired satellite
        
        Returns:
            Cartesian orbit of desired satellite at epoch
        """
        propagator = TLEPropagator.selectExtrapolator(sat_tle)
        epoch = sat_tle.getDate()
        pv = propagator.getPVCoordinates(epoch, self.eme2000)
        initial_orbit = CartesianOrbit(pv, self.eme2000, epoch, Constants.WGS84_EARTH_MU)

        return initial_orbit

    def propagateTLE(self, sat, prop_dur, prop_res):
        """
        Propagates the satellite's orbit using SGP4/SDP4

        Args:
            sat: string
                The ID of the desired satellite
            
            prop_dur: float
                For how many days forward should the orbit be propagated
            
            prop_res: float
                How many seconds between PV updates

        Returns:
            Array of position and velocity coordinates
        """
            
        extrap_date = self.time_now
        propagator = TLEPropagator.selectExtrapolator(self.sat_data.get(sat)['TLE'])
        pvs = []
        final_date = extrap_date.shiftedBy(60 * 60 * 24 * prop_dur) #seconds

        while (extrap_date.compareTo(final_date) <= 0.0):
            pvs.append(propagator.getPVCoordinates(extrap_date, self.eme2000))
            extrap_date = extrap_date.shiftedBy(prop_res)
        
        return pvs

    def propagateNumerical(self, sat, prop_dur, prop_res):
        """
        Propagates the satellite's orbit using numerical propagation

        Args:
            sat: string
                The ID of the desired satellite
            
            prop_dur: float
                For how many days forward should the orbit be propagated
            
            prop_res: float
                How many seconds between PV updates

        Returns:
            Array of position and velocity coordinates
        """

        initial_orbit = self.initialOrbitTLE(self.sat_data.get(sat)['TLE']) # Defines Cartesian orbit using latest TLE data
        sat_mass = 100.0 # kg
        initial_state = SpacecraftState(initial_orbit, sat_mass)

        min_step = 0.001
        max_step = 1000.0
        init_step = 60.0
        position_tolerance = 1.0
        orbit_type = OrbitType.CARTESIAN
        tol = NumericalPropagator.tolerances(position_tolerance, initial_orbit, orbit_type)

        integrator = DormandPrince853Integrator(min_step, max_step, 
                                                JArray_double.cast_(tol[0]),  # Double array of doubles needs to be casted in Python
                                                JArray_double.cast_(tol[1]))
        integrator.setInitialStepSize(init_step)

        propagator_num = NumericalPropagator(integrator)
        propagator_num.setOrbitType(orbit_type)
        propagator_num.setInitialState(initial_state)

        gravityProvider = GravityFieldFactory.getNormalizedProvider(8, 8)
        propagator_num.addForceModel(HolmesFeatherstoneAttractionModel(self.earth(self.itrf).getBodyFrame(), gravityProvider))

        start_date = self.time_now

        t = [start_date.shiftedBy(float(dt)) for dt in np.arange(0, prop_dur * 86400, prop_res)]
        pvs = [propagator_num.propagate(tt).getPVCoordinates() for tt in t]

        return pvs