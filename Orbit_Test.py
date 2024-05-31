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

from Orbitography_Functions import *

##############################################################################################################

now = datetime.now(UTC)
time_now = AbsoluteDate(now.year, now.month, now.day, now.hour, now.minute + 1, 0.0, TimeScalesFactory.getUTC())

of = OrbitographyFunctions(sat_data=sat_data, time_now=time_now)

# Define Earth and inertial reference frames
eme2000 = FramesFactory.getEME2000()
itrf    = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

id_st = os.getenv("id_st")
pass_st = os.getenv("pass_st")
st = SpaceTrackClient(identity=id_st, password=pass_st)

prop_data, trace_orbit = {},[]

for sat in sat_data:
    tle = st.tle(norad_cat_id=sat_data[sat]['norad_id'], epoch='<{}'.format(time_now), orderby='epoch desc', limit=1, format='tle').split('\n')
    sat_data[sat]['TLE'] = TLE(tle[0],tle[1])

    #pvs = propagateTLE(sat, prop_dur=1.0, prop_res=60.0)
    pvs = of.propagateNumerical(sat, prop_dur=.25, prop_res=60.0)

    # Extracts position-velocity data and creates database of calculated values
    prop_data[sat]                = pd.DataFrame(data=pvs, columns=['pv'])
    prop_data[sat]['Position']    = prop_data[sat]['pv'].apply(lambda x: x.getPosition())
    prop_data[sat]['x']           = prop_data[sat]['Position'].apply(lambda pos: pos.x)
    prop_data[sat]['y']           = prop_data[sat]['Position'].apply(lambda pos: pos.y)
    prop_data[sat]['z']           = prop_data[sat]['Position'].apply(lambda pos: pos.z)
    prop_data[sat]['Radius']      = prop_data[sat]['Position'].apply(lambda pos: ((pos.x)**2 + (pos.y)**2 + (pos.z)**2)**0.5)
    prop_data[sat]['datetime']    = prop_data[sat]['pv'].apply(lambda x: absolutedate_to_datetime(x.getDate()))
    prop_data[sat]['groundpoint'] = prop_data[sat]['pv'].apply(lambda pv: of.earth(eme2000).transform(pv.position, eme2000, pv.date))
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
                                dt.strftime('%Y-%m-%d %H:%M:%S'), 'UTC', of.getEpochDelta(sat)
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

trace_earth = of.plotEarth()
trace_background = of.plotBackground()

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