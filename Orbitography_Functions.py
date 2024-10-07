import orekit
vm = orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir, datetime_to_absolutedate
setup_orekit_curdir()

from org.orekit.frames import FramesFactory # type: ignore
from org.orekit.bodies import OneAxisEllipsoid # type: ignore
from org.orekit.time import TimeScalesFactory, AbsoluteDate # type: ignore
from org.orekit.utils import IERSConventions, Constants # type: ignore
from org.orekit.propagation.analytical.tle import TLE, TLEPropagator # type: ignore
from org.orekit.propagation.numerical import NumericalPropagator # type: ignore
from org.orekit.propagation import SpacecraftState  # type: ignore
from org.orekit.orbits import OrbitType, CartesianOrbit # type: ignore
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator # type: ignore
from org.orekit.forces.gravity.potential import GravityFieldFactory # type: ignore
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel # type: ignore
from orekit import JArray_double

from java.util import Arrays # type: ignore
from java.io import File # type: ignore

import pandas as pd
import numpy as np
import os
import json
import requests
import time
import plotly.graph_objs as go

from datetime import datetime, timedelta, UTC
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

now_UTC     = datetime.now(UTC)
now_UTC     = datetime_to_absolutedate(now_UTC - timedelta(seconds=now_UTC.second, microseconds=now_UTC.microsecond))

class DatabaseFunctions:
    def __init__(self):
        with open('Metadata.json', 'r') as file:
            self.path = json.load(file)['path']

    def downloadEphemerides(self, date_range='>now-30'):
        '''
        Downloads the latest ephemerides within the requested time range

        Args:
            date_range: string
                The requested updated period, formatted to Space-Track's API. Defaults to the previous 30 days
        
        Returns:
            -1:
                Indicates that there is no new ephemerides data, signals to terminate database update
            response: list[dict]
                A list of for all ephemerides for objects in orbit that have been updated in the requested time period
        '''

        if (response := self.spaceTrackRequest(f'/class/gp/decay_date/null-val/epoch/{date_range}/orderby/norad_cat_id/format/json')) == []:
            return -1
        else:
            with open(self.path['ephemerides'], 'w') as file:
                json.dump(response, file)
            return response
        
    def spaceTrackRequest(self, requestQuery):
        '''
        Sends a query to Space-Track.org and returns the response

        Args:
            requestQuery: string
                The desired Space-Track API query
        
        Returns:
            response.text: list[dict]
                A list of for all ephemerides for objects in orbit that have been updated in the requested time period, loaded from the json object received from Space-Track
        '''

        uriBase                 = 'https://www.space-track.org'
        requestLogin            = '/ajaxauth/login'
        requestCmdAction        = '/basicspacedata/query'

        id_st   = os.getenv('id_st')
        pass_st = os.getenv('pass_st')

        with requests.Session() as session:
            if (login_response := session.post(uriBase + requestLogin, data = {'identity': id_st, 'password': pass_st})).status_code != 200:
                raise Exception(f'{login_response.status_code}\n{login_response.text}')
            
            if (response := session.get(uriBase + requestCmdAction + requestQuery)).status_code != 200:
                raise Exception(f'{response.status_code}\n{response.text}')

        return json.loads(response.text)


class PlotFunctions:
    def __init__(self, prop_data_file):
        self.prop_data = pd.read_json(prop_data_file)

    def createSpheroid(self, texture, equatorial_radius, flattening=0):
        '''
        Creates a spheroid

        Args:
            texture: 

            equatorial_radius: float
                The radius of the spheroid at the equator, in metres
            
            flattening: float, optional
                The flattening value of the spheroid. Leave blank to define a perfect sphere

        Returns:
            Array of radii in the x, y, and z directions, where z is up. Units in metres
        '''

        N_lat = int(texture.shape[0])
        N_lon = int(texture.shape[1])
        theta = np.linspace(0,2*np.pi,N_lat)
        phi   = np.linspace(0,np.pi,N_lon) 

        axes_radii = [equatorial_radius * np.outer(np.cos(theta),np.sin(phi)),
                      equatorial_radius * np.outer(np.sin(theta),np.sin(phi)),
                      equatorial_radius * (1 - flattening) * np.outer(np.ones(N_lat),np.cos(phi))]
        
        return axes_radii

    def plotEarth(self):
        '''
        Creates a spherical image of the Earth
        '''

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
        '''
        Creates a black spherical background
        '''

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
        '''
        Calculates time between the epoch of the current satellite and time_now

        Args:
            sat: the ID of the desired satellite
        Returns:
            TimeDelta of time since epoch
        '''
        epoch = datetime_to_absolutedate(datetime.strptime(self.prop_data.loc[sat, 'epoch'], '%Y/%m/%d %H:%M:%S.%f'))
        return timedelta(seconds = now_UTC.durationFrom(epoch))

    def plotOrbits(self, metadata_file, limit=0):
        '''
        Creates a 3D orbital plot
        '''

        with open(metadata_file, 'r') as file:
            metadata = json.load(file)

        trace_orbit = []
        rendered_prop_data = self.prop_data[self.prop_data['object_type'].isin([key for key, value in metadata['objects'].items() if value])]
        if limit:
            rendered_prop_data = rendered_prop_data.iloc[:limit]
        dts         = metadata['datetimes']

        lat_lambda  = lambda x: 'N' if x >= 0 else 'S'
        lon_lambda  = lambda x: 'E' if x >= 0 else 'W'

        n       = 0
        length  = len(rendered_prop_data)
        for sat in rendered_prop_data.index:
            lats, lons, rs = rendered_prop_data.loc[sat, ['latitude', 'longitude', 'radius']]
            epoch = self.getEpochDelta(sat)
            text = [(                                                                                                       # Defines label text when hovering over orbits
                f'{rendered_prop_data.loc[sat, 'object_name']} ({sat})<br>'                                                 # Satellite name and NORAD CAT ID
                f'{'-'*56}<br>'                                                                                             # -----
                f'{int(abs(lat)):02}° {int(abs(lat)%1*60):02}\' {abs(lat)%1*3600%60:07.4f}\" {lat_lambda(lat)}, '           # Latitude (° ' ')
                f'{int(abs(lon)):02}° {int(abs(lon)%1*60):02}\' {abs(lon)%1*3600%60:07.4f}\" {lon_lambda(lon)}<br>'         # Longitude (° ' ')
                f'Radius (Alt): {r/1000:.2f} km ({(r - Constants.WGS84_EARTH_EQUATORIAL_RADIUS)/1000:.2f} km)<br>'          # Orbital radius & height (km)
                f'{'-'*56}<br>'                                                                                             # -----
                f'{dt} UTC<br>'                                                                                             # Date & time (UTC)
                f'{epoch} since epoch'                                                                                      # Time since epoch TODO: Format time to DD days HHh, MMm, SSs
                ) for lat, lon, r, dt in zip(lats, lons, rs, dts)
                ]

            trace_orbit.append(go.Scatter3d(x               = rendered_prop_data.loc[sat, 'x'],
                                            y               = rendered_prop_data.loc[sat, 'y'],
                                            z               = rendered_prop_data.loc[sat, 'z'],
                                            marker          = dict(size=0.5),
                                            line            = dict(color='white',width=1),
                                            hoverinfo       = 'text',
                                            hovertemplate   = '%{text}<extra></extra>',
                                            text            = text
                                            ))
            n += 1
            if not n % 50:
                print(f'Tracing orbits:  {n/length:06.2%}')
        print('Tracing orbits: 100.00%')

        trace_earth = self.plotEarth()
        trace_background = self.plotBackground()

        layout = go.Layout(paper_bgcolor='black',
                           scene = dict(xaxis = dict(visible=False),
                                        yaxis = dict(visible=False),
                                        zaxis = dict(visible=False),
                                        hovermode = 'closest',
                                        ),
                           showlegend = False
                           )

        fig = go.Figure(data=[trace_earth, trace_background] + trace_orbit, 
                        layout=layout
                        )
        fig.show()

class SatelliteFunctions:

    def __init__(self, **kwargs):
        self.sat_data   = kwargs.get('sat_data')
        self.eme2000    = FramesFactory.getEME2000()
        self.itrf       = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    def earth(self, frame):
        return OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, 
                                Constants.WGS84_EARTH_FLATTENING, 
                                frame)
    
    def toTLE(self, sat_tle_tuple):
        return TLE(sat_tle_tuple[0], sat_tle_tuple[1])

    def initialOrbitTLE(self, sat_tle):
        '''
        Initialises a Cartesian orbit of an object, using TLE data

        Args:
            sat_tle: 
                TLE data of desired satellite
        
        Returns:
            Cartesian orbit of desired satellite at epoch
        '''
        propagator      = TLEPropagator.selectExtrapolator(sat_tle)
        epoch           = sat_tle.getDate()
        pv              = propagator.getPVCoordinates(epoch, self.eme2000)
        initial_orbit   = CartesianOrbit(pv, self.eme2000, epoch, Constants.WGS84_EARTH_MU)

        return initial_orbit

    def propagateTLE(self, sat, resolution, **kwargs):
        '''
        Propagates the satellite's orbit using SGP4/SDP4 propagation

        Args:
            sat: string
                The NORAD ID of the desired satellite
            
            resolution: float
                How many seconds between PV coordinate updates
            
            **kwargs:
                Provide either duration or start AND end

                    duration: int or float
                        Number of days from now_UTC to propagate the orbit

                    start: AbsoluteDate
                        The start date to propagate the orbit from
                    
                    end: AbsoluteDate
                        The target date to propagate the orbit until

        Returns:
            list[TimeStampedPVCoordinates]
                Array of time-stamped position and velocity coordinates
        '''

        start_date  = kwargs.get('start')
        end_date    = kwargs.get('end')
        duration    = kwargs.get('duration')
        
        if not(bool(duration) ^ bool(start_date and end_date)):
            raise Exception('Provide either only a duration or both a start and end date')
        elif not start_date:
            start_date  = now_UTC
            duration    = duration * 86400.
        elif not duration:
            duration    = end_date.shiftedBy(.1).durationFrom(start_date)

        propagator  = TLEPropagator.selectExtrapolator(self.toTLE(self.sat_data.loc[sat, 'TLE']))
        t           = [start_date.shiftedBy(float(dt)) for dt in np.arange(0, duration, resolution)]
        pvs         = [propagator.propagate(tt).getPVCoordinates() for tt in t]

        return pvs

    def propagateNumerical(self, sat, resolution, **kwargs): # TODO: Update sat_data from dictionary to dataframe
        '''
        Propagates the satellite's orbit using numerical propagation

        Args:
            sat: string
                The NORAD ID of the desired satellite
            
            resolution: int or float
                How many seconds between PV coordinate updates
            
            **kwargs:
                Provide either duration or start AND end

                    duration: float
                        Number of days from now_UTC to propagate the orbit

                    start: AbsoluteDate
                        The start date to propagate the orbit from
                    
                    end: AbsoluteDate
                        The target date to propagate the orbit until

        Returns:
            pvs: list[TimeStampedPVCoordinates]
                Array of time-stamped position and velocity coordinates
        '''

        start_date  = kwargs.get('start')
        end_date    = kwargs.get('end')
        duration    = kwargs.get('duration')

        if not(bool(duration) ^ bool(start_date and end_date)):
            raise Exception('Provide either only a duration or both a start and end date')
        elif not start_date:
            start_date  = now_UTC
            duration    = duration * 86400.
        elif not duration:
            duration    = end_date.shiftedBy(.1).durationFrom(start_date)

        initial_orbit   = self.initialOrbitTLE(self.toTLE(self.sat_data[sat]['TLE'])) # Defines initial Cartesian orbit using latest TLE data
        sat_mass        = self.sat_data[sat]['mass']
        initial_state   = SpacecraftState(initial_orbit, sat_mass)

        min_step        = 0.001
        max_step        = 1000.0
        init_step       = 60.0
        pos_tolerance   = 1.0
        orbit_type      = OrbitType.CARTESIAN
        tol             = NumericalPropagator.tolerances(pos_tolerance, initial_orbit, orbit_type)

        integrator = DormandPrince853Integrator(min_step, max_step, 
                                                JArray_double.cast_(tol[0]),  # Double array of doubles needs to be casted in Python
                                                JArray_double.cast_(tol[1]))
        integrator.setInitialStepSize(init_step)

        propagator = NumericalPropagator(integrator)
        propagator.setOrbitType(orbit_type)
        propagator.setInitialState(initial_state)

        gravityProvider = GravityFieldFactory.getNormalizedProvider(8, 8)
        propagator.addForceModel(HolmesFeatherstoneAttractionModel(self.itrf, gravityProvider))

        t   = [start_date.shiftedBy(float(dt)) for dt in np.arange(0, duration, resolution)]
        pvs = [propagator.propagate(tt).getPVCoordinates() for tt in t]

        return pvs
    
    def propagateNumerical2(self, sat, resolution, **kwargs): # TODO: Update sat_data from dictionary to dataframe
        '''
        Propagates the satellite's orbit using numerical propagation

        Args:
            sat: string
                The NORAD ID of the desired satellite
            
            resolution: float
                How many seconds between PV coordinate updates
            
            **kwargs:
                Provide either duration or start AND end

                    duration: int or float
                        Number of days from now_UTC to propagate the orbit

                    start: AbsoluteDate
                        The start date to propagate the orbit from
                    
                    end: AbsoluteDate
                        The target date to propagate the orbit until

        Returns:
            pvs: list[TimeStampedPVCoordinates]
                Array of time-stamped position and velocity coordinates
        '''

        start_date  = kwargs.get('start')
        end_date    = kwargs.get('end')
        duration    = kwargs.get('duration')

        if not(bool(duration) ^ bool(start_date and end_date)):
            raise Exception('Provide either only a duration or both a start and end date')
        elif not start_date:
            start_date  = now_UTC
            duration    = duration * 86400.
        elif not duration:
            duration    = end_date.shiftedBy(.1).durationFrom(start_date)

        initial_orbit   = self.initialOrbitTLE(self.toTLE(self.sat_data[sat]['TLE'])) # Defines initial Cartesian orbit using latest TLE data
        sat_mass        = self.sat_data[sat]['mass']
        initial_state   = SpacecraftState(initial_orbit, sat_mass)

        min_step        = 0.001
        max_step        = 1000.0
        init_step       = 60.0
        pos_tolerance   = 1.0
        orbit_type = OrbitType.CARTESIAN
        tol = NumericalPropagator.tolerances(pos_tolerance, initial_orbit, orbit_type)

        integrator = DormandPrince853Integrator(min_step, max_step, 
                                                JArray_double.cast_(tol[0]),  # Double array of doubles needs to be casted in Python
                                                JArray_double.cast_(tol[1]))
        integrator.setInitialStepSize(init_step)

        propagator = NumericalPropagator(integrator)
        propagator.setOrbitType(orbit_type)
        propagator.setInitialState(initial_state)

        gravityProvider = GravityFieldFactory.getNormalizedProvider(8, 8)
        propagator.addForceModel(HolmesFeatherstoneAttractionModel(self.itrf, gravityProvider))

        t   = [start_date.shiftedBy(float(dt)) for dt in np.arange(0, duration, resolution)]
        pvs = [propagator.propagate(tt).getPVCoordinates() for tt in t]

        return pvs