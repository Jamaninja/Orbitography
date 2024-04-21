def create_Spheroid(texture, equatorial_radius, flattening=0):
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

    if not 0 <= flattening < 1:
	    raise Exception('Flattening value must be between 0 (inclusive) and 1 (exclusive)')

    N_lat = int(texture.shape[0])
    N_lon = int(texture.shape[1])
    theta = np.linspace(0,2*np.pi,N_lat)
    phi   = np.linspace(0,np.pi,N_lon) 

    axes_radii = [equatorial_radius * np.outer(np.cos(theta),np.sin(phi)),
                  equatorial_radius * np.outer(np.sin(theta),np.sin(phi)),
                  equatorial_radius * (1 - flattening) * np.outer(np.ones(N_lat),np.cos(phi))
                  ]
    return axes_radii

def plot_Earth(texture):
    axes = create_Spheroid(texture, Constants.IERS2010_EARTH_EQUATORIAL_RADIUS, Constants.IERS2010_EARTH_FLATTENING)
    surf = go.Surface(x=axes[0], y=axes[1], z=axes[2],
                      surfacecolor = texture,
                      colorscale   = colorscale,
                      showscale    = False,
                      hoverinfo    = 'skip',
                      )    
    return surf

def plot_Background(texture):
    axes = create_Spheroid(texture, 100*Constants.IERS2010_EARTH_EQUATORIAL_RADIUS)
    surf = go.Surface(x=axes[0], y=axes[1], z=axes[2],
                      colorscale = [[0.0, 'black'],
                                    [1.0, 'black']],
                      showscale  = False,
                      hoverinfo  = 'skip',
                      )    
    return surf

def getNow():
    """
    Fetches the current time, rounded down to the minute

    Args:
        None
    Returns:
        AbsoluteDate of current time
    """

    now = datetime.now(UTC)
    return AbsoluteDate(now.year, now.month, now.day, now.hour, now.minute, 0.0, TimeScalesFactory.getUTC())

def getEpochDelta(sat):
    """
    Calculates time between the epoch of the current satellite and the current time as called by getNow()

    Args:
        sat: the ID of the desired satellite
    Returns:
        TimeDelta of time since epoch
    """

    epoch = sat_data.get(sat)['TLE'].getDate()
    return timedelta(seconds = getNow().durationFrom(epoch))

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
from java.io import File # type: ignore

import pandas as pd
import numpy as np
from satellite_tle import fetch_all_tles, fetch_latest_tles
from datetime import datetime, timedelta, UTC

from PIL import Image
import plotly.offline as px
import plotly.graph_objs as go

#Data required for rendering the Earth's texture, and rotates the planet to align prime meridian
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

#Fetches TLE data for a list of satellites
norad_ids = [25544, 20580, 26590, 32253, 40697, 42063, 37820, 25338, 28654, 33591, 57479, 56757, 56174]
sat_data, tles = {}, fetch_latest_tles(norad_ids)

for norad_id, (source, tle) in tles.items():
    sat_id = tle[1].split(" ")[2]
    sat_data[sat_id] = {'TLE': TLE(tle[1],tle[2]), 'Name': tle[0]}

#Define Earth and inertial reference frame
#inertial_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, True) #Rotating inertial reference frame
inertial_frame = FramesFactory.getEME2000() #Nonrotating inertial reference frame
earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, 
                         Constants.WGS84_EARTH_FLATTENING, 
                         inertial_frame)

###############################################################################################################
prop_data, trace_orbit = {},[]
propagation_days = 0.25 #days, float
propagation_resolution = 60.0 #seconds, float

def TLE_Propagator(sat, prop_days, prop_res):
    extrap_date = getNow()
    propagator = TLEPropagator.selectExtrapolator(sat_data.get(sat)['TLE'])
    pvs = []
    final_date = extrap_date.shiftedBy(60 * 60 * 24 * prop_days) #seconds

    while (extrap_date.compareTo(final_date) <= 0.0):
        pvs.append(propagator.getPVCoordinates(extrap_date, inertial_frame))
        extrap_date = extrap_date.shiftedBy(prop_res)
    
    return pvs

for sat in sat_data:
    pvs = TLE_Propagator(sat, propagation_days, propagation_resolution)

    #Extracts position-velocity data and fills out database of calculated values
    prop_data[sat] = pd.DataFrame(data=pvs, columns=['pv'])
    prop_data[sat]['Position']    = prop_data[sat]['pv'].apply(lambda x: x.getPosition())
    prop_data[sat]['x']           = prop_data[sat]['Position'].apply(lambda pos: pos.x)
    prop_data[sat]['y']           = prop_data[sat]['Position'].apply(lambda pos: pos.y)
    prop_data[sat]['z']           = prop_data[sat]['Position'].apply(lambda pos: pos.z)
    prop_data[sat]['Radius']      = prop_data[sat]['Position'].apply(lambda pos: ((pos.x)**2 + (pos.y)**2 + (pos.z)**2)**0.5)
    prop_data[sat]['datetime']    = prop_data[sat]['pv'].apply(lambda x: absolutedate_to_datetime(x.getDate()))
    prop_data[sat]['groundpoint'] = prop_data[sat]['pv'].apply(lambda pv: earth.transform(pv.position, inertial_frame, pv.date))
    prop_data[sat]['latitude']    = np.degrees(prop_data[sat].groundpoint.apply(lambda gp: gp.latitude))
    prop_data[sat]['longitude']   = np.degrees(prop_data[sat].groundpoint.apply(lambda gp: gp.longitude))
    
    #Defines label text when hovering over orbits and creates orbit traces
        #Satellite name
        #Latitude & longitude
        #Date & time (UTC)

    text = []
    name = sat_data.get(sat)['Name']
    lat_lambda = lambda x: 'N' if x >= 0 else 'S'
    lon_lambda = lambda x: 'E' if x >= 0 else 'W'

    for lat, lon, r, dt in zip(prop_data[sat]['latitude'], prop_data[sat]['longitude'], prop_data[sat]['Radius'], prop_data[sat]['datetime']):    
        text.append('''{}<br>{}
<br>{:02}° {:02}\' {:02.4f}\" {},
{:02}° {:02}\' {:02.4f}\" {}
<br>Radius (Alt): {:.2f} km ({:.2f} km)<br>{}
<br>{} {}<br>epoch+{}'''.format(name, '-'*56,
                            int(abs(lat)), int(abs(lat)%1*60), abs(lat)%1*3600%60, lat_lambda(lat),
                            int(abs(lon)), int(abs(lon)%1*60), abs(lon)%1*3600%60, lon_lambda(lon), 
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
proximity_bound = 10e3 #meters
#Detecting collisions
    #First, determine which satellites are within a 10km box
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
        #calculate distance between sat_a and sat_b
        print()
'''

trace_earth = plot_Earth(earth_texture)
trace_background = plot_Background(earth_texture)

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