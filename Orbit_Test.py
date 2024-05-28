def createSpheroid(texture, equatorial_radius, flattening=0):
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

def plotEarth(texture):
    """
    Creates a spherical image of the Earth
    """

    axes = createSpheroid(texture, Constants.IERS2010_EARTH_EQUATORIAL_RADIUS, Constants.IERS2010_EARTH_FLATTENING)
    surf = go.Surface(x=axes[0], y=axes[1], z=axes[2],
                      surfacecolor = texture,
                      colorscale   = colorscale,
                      showscale    = False,
                      hoverinfo    = 'skip',
                      )    
    return surf

def plotBackground(texture):
    """
    Creates a black spherical background
    """

    axes = createSpheroid(texture, 100*Constants.IERS2010_EARTH_EQUATORIAL_RADIUS)
    surf = go.Surface(x=axes[0], y=axes[1], z=axes[2],
                      colorscale = [[0.0, 'black'],
                                    [1.0, 'black']],
                      showscale  = False,
                      hoverinfo  = 'skip',
                      )    
    return surf

def getEpochDelta(sat):
    """
    Calculates time between the epoch of the current satellite and time_now

    Args:
        sat: the ID of the desired satellite
    Returns:
        TimeDelta of time since epoch
    """

    epoch = sat_data.get(sat)['TLE'].getDate()
    return timedelta(seconds = time_now.durationFrom(epoch))

def propagateTLE(sat, prop_dur, prop_res):
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
        
    extrap_date = time_now
    propagator = TLEPropagator.selectExtrapolator(sat_data.get(sat)['TLE'])
    pvs = []
    final_date = extrap_date.shiftedBy(60 * 60 * 24 * prop_dur) #seconds

    while (extrap_date.compareTo(final_date) <= 0.0):
        pvs.append(propagator.getPVCoordinates(extrap_date, eme2000))
        extrap_date = extrap_date.shiftedBy(prop_res)
    
    return pvs

def initialOrbitTLE(sat_tle):
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
    pv = propagator.getPVCoordinates(epoch, eme2000)
    initial_orbit = CartesianOrbit(pv, eme2000, epoch, Constants.WGS84_EARTH_MU)

    return initial_orbit

def propagateNumerical(sat, prop_dur, prop_res):
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

    initial_orbit = initialOrbitTLE(sat_data.get(sat)['TLE']) # Defines Cartesian orbit using latest TLE data
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
    propagator_num.addForceModel(HolmesFeatherstoneAttractionModel(earth(itrf).getBodyFrame(), gravityProvider))

    start_date = time_now

    t = [start_date.shiftedBy(float(dt)) for dt in np.arange(0, prop_dur * 86400, prop_res)]
    pvs = [propagator_num.propagate(tt).getPVCoordinates() for tt in t]

    return pvs

##############################################################################################################

sat_data = {
    'ENVISAT': {
        'norad_id': 27386,  # For Space-Track TLE queries
        'cospar_id': '0200901',  # For laser ranging data queries
        'sic_id': '6179',  # For writing in CPF files
        'mass': 8000.0, # kg; TODO: compute proper value
        'cross_section': 100.0, # m2; TODO: compute proper value
        'cd': 2.0, # TODO: compute proper value
        'cr': 1.0  # TODO: compute proper value
    },
    'LAGEOS2': {
        'norad_id': 22195,
        'cospar_id': '9207002',
        'sic_id': '5986',
        'mass': 405.0, # kg
        'cross_section': 0.2827, # m2
        'cd': 2.0, # TODO: compute proper value
        'cr': 1.0  # TODO: compute proper value
    },
    'TECHNOSAT': {
        'norad_id': 42829,
        'cospar_id': '1704205',
        'sic_id': '6203',
        'mass': 20.0, # kg
        'cross_section': 0.10, # m2,
        'cd': 2.0, # TODO: compute proper value
        'cr': 1.0  # TODO: compute proper value
    },
    'SNET1': {
        'norad_id': 43189,
        'cospar_id': '1801410',
        'sic_id': '6204',
        'mass': 8.0, # kg
        'cross_section': 0.07, # m2
        'cd': 2.0, # TODO: compute proper value
        'cr': 1.0  # TODO: compute proper value
    },
    'ISS (ZARYA)': {
        'norad_id': 25544,
        'cospar_id': '1998-067A',
        'sic_id': '',
        'mass': 450_000.0, # kg
        'cross_section': 2_481.0, # m2
        'cd': 2.0, # TODO: compute proper value
        'cr': 1.0  # TODO: compute proper value
    }
}

##############################################################################################################

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

import pandas as pd
import numpy as np
from spacetrack import SpaceTrackClient
from datetime import datetime, timedelta, UTC
from PIL import Image
import plotly.graph_objs as go

from dotenv import load_dotenv
import os
load_dotenv()

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

##############################################################################################################

now = datetime.now(UTC)
time_now = AbsoluteDate(now.year, now.month, now.day, now.hour, now.minute + 1, 0.0, TimeScalesFactory.getUTC())

# Define Earth and inertial reference frames
eme2000 = FramesFactory.getEME2000()
itrf    = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

def earth(frame):
    return OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, 
                            Constants.WGS84_EARTH_FLATTENING, 
                            frame)

id_st = os.getenv("id_st")
pass_st = os.getenv("pass_st")
st = SpaceTrackClient(identity=id_st, password=pass_st)

prop_data, trace_orbit = {},[]

for sat in sat_data:
    tle = st.tle(norad_cat_id=sat_data[sat]['norad_id'], epoch='<{}'.format(time_now), orderby='epoch desc', limit=1, format='tle').split('\n')
    sat_data[sat]['TLE'] = TLE(tle[0],tle[1])

    #pvs = propagateTLE(sat, prop_dur=1.0, prop_res=60.0)
    pvs = propagateNumerical(sat, prop_dur=.25, prop_res=60.0)

    # Extracts position-velocity data and creates database of calculated values
    prop_data[sat]                = pd.DataFrame(data=pvs, columns=['pv'])
    prop_data[sat]['Position']    = prop_data[sat]['pv'].apply(lambda x: x.getPosition())
    prop_data[sat]['x']           = prop_data[sat]['Position'].apply(lambda pos: pos.x)
    prop_data[sat]['y']           = prop_data[sat]['Position'].apply(lambda pos: pos.y)
    prop_data[sat]['z']           = prop_data[sat]['Position'].apply(lambda pos: pos.z)
    prop_data[sat]['Radius']      = prop_data[sat]['Position'].apply(lambda pos: ((pos.x)**2 + (pos.y)**2 + (pos.z)**2)**0.5)
    prop_data[sat]['datetime']    = prop_data[sat]['pv'].apply(lambda x: absolutedate_to_datetime(x.getDate()))
    prop_data[sat]['groundpoint'] = prop_data[sat]['pv'].apply(lambda pv: earth(eme2000).transform(pv.position, eme2000, pv.date))
    prop_data[sat]['latitude']    = np.degrees(prop_data[sat].groundpoint.apply(lambda gp: gp.latitude))
    prop_data[sat]['longitude']   = np.degrees(prop_data[sat].groundpoint.apply(lambda gp: gp.longitude))
    
    # Defines label text when hovering over orbits and creates orbit traces
        # Satellite name
        # Latitude & longitude
        # Date & time (UTC)
        # Time since epoch

    text = []
    name = sat
    lat_lambda = lambda x: 'N' if x >= 0 else 'S'
    lon_lambda = lambda x: 'E' if x >= 0 else 'W'

    for lat, lon, r, dt in zip(prop_data[sat]['latitude'], prop_data[sat]['longitude'], prop_data[sat]['Radius'], prop_data[sat]['datetime']):    
        text.append('''{}<br>{}
<br>{:02}° {:02}\' {:02.4f}\" {}, {:02}° {:02}\' {:02.4f}\" {}
<br>Radius (Alt): {:.2f} km ({:.2f} km)<br>{}
<br>{} {}<br>epoch+{}'''.format(name, '-'*56,
                                int(abs(lat)), int(abs(lat)%1*60), abs(lat)%1*3600%60, lat_lambda(lat), int(abs(lon)), int(abs(lon)%1*60), abs(lon)%1*3600%60, lon_lambda(lon), 
                                r/1000, (r - Constants.WGS84_EARTH_EQUATORIAL_RADIUS)/1000,'-'*56,
                                dt.strftime('%Y-%m-%d %H:%M:%S'), 'UTC', getEpochDelta(sat)
                                )
                    )

    trace_orbit.append(go.Scatter3d(x=[x for x in prop_data[sat]['x']],
                                    y=[y for y in prop_data[sat]['y']],
                                    z=[z for z in prop_data[sat]['z']],
                                    marker=dict(size=0.5),
                                    line=dict(color='white',width=1),
                                    hoverinfo='text',
                                    hovertemplate='%{text}<extra></extra>',
                                    text = text
                                    )
                       )

'''
proximity_bound = 10e3 # meters
# Detecting collisions
    # First, determine which satellites are within a 10km box
while True:
    a,b = 0,1
    sat_a, sat_b = sat_data[a], sat_data[b]
    if abs(prop_data[sat_a]['x'] - prop_data[sat_b]['x']) > proximity_bound:
        break
    elif abs(prop_data[sat_a]['y'] - prop_data[sat_b]['y']) > proximity_bound:
        break
    elif abs(prop_data[sat_a]['z'] - prop_data[sat_b]['z']) > proximity_bound:
        break
    else:
        # Calculate distance between sat_a and sat_b
        print()
'''

trace_earth = plotEarth(earth_texture)
trace_background = plotBackground(earth_texture)

layout = go.Layout(paper_bgcolor='black',
                   scene = dict(xaxis = dict(visible=False),
                                yaxis = dict(visible=False),
                                zaxis = dict(visible=False),
                                hovermode = 'closest',
                                ),
                   showlegend = False
                   )

fig = go.Figure(data=[trace_earth, trace_background] + trace_orbit, 
                layout=layout,
                )
fig.show()