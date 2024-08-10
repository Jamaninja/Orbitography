# Code adapted from GorgiAstro on github:
# https://github.com/GorgiAstro/laser-orbit-determination/blob/master/02-orbit-determination-example.ipynb

sat_list = {
    'envisat': {
        'norad_id': 27386,  # For Space-Track TLE queries
        'cospar_id': '0200901',  # For laser ranging data queries
        'sic_id': '6179',  # For writing in CPF files
        'mass': 8000.0, # kg; TODO: compute proper value
        'cross_section': 100.0, # m2; TODO: compute proper value
        'cd': 2.0, # TODO: compute proper value
        'cr': 1.0  # TODO: compute proper value
    },
    'lageos2': {
        'norad_id': 22195,
        'cospar_id': '9207002',
        'sic_id': '5986',
        'mass': 405.0, # kg
        'cross_section': 0.2827, # m2
        'cd': 2.0, # TODO: compute proper value
        'cr': 1.0  # TODO: compute proper value
    },
    'technosat': {
        'norad_id': 42829,
        'cospar_id': '1704205',
        'sic_id': '6203',
        'mass': 20.0, # kg
        'cross_section': 0.10, # m2,
        'cd': 2.0, # TODO: compute proper value
        'cr': 1.0  # TODO: compute proper value
    },
    'snet1': {
        'norad_id': 43189,
        'cospar_id': '1801410',
        'sic_id': '6204',
        'mass': 8.0, # kg
        'cross_section': 0.07,
        'cd': 2.0, # TODO: compute proper value
        'cr': 1.0  # TODO: compute proper value
    }
}

sc_name = 'lageos2'

"""
NPT: Normal point data. Recommended option. The data is pre-filtered by the laser data providers
FRD: Full-rate data. Warning, these are a lot of data points (potentially tens of thousands per day),
    the execution time could be greatly increased
"""
laser_data_type = 'FRD'

import numpy as np
import pandas as pd
from orekit.pyhelpers import datetime_to_absolutedate, absolutedate_to_datetime
from datetime import datetime, timedelta
from Orbitography_Functions import SatelliteFunctions, now

#sf = SatelliteFunctions()

range_weight = 1.0 # Will be normalized later (i.e divided by the number of observations)
range_sigma = 1.0 # Estimated covariance of the range measurements, in meters

az_weight = 0.1  # Do not weigh the Az/El measurements too much because they are much less accurate than ranges
el_weight = 0.1
az_sigma = float(np.deg2rad(0.01))
el_sigma = float(np.deg2rad(0.01))

odDate = now # Beginning of the orbit determination
collectionDuration = 2 # days
startCollectionDate = odDate.shiftedBy(collectionDuration*-86400.)

# Orbit propagator parameters
prop_min_step = 0.001 # s
prop_max_step = 300.0 # s
prop_position_error = 10.0 # m

# Estimator parameters
estimator_position_scale = 1.0 # m
estimator_convergence_thres = 1e-2
estimator_max_iterations = 25
estimator_max_evaluations = 35

# Credentials

# Space-Track log in
from dotenv import load_dotenv
import os
load_dotenv()

import spacetrack.operators as op
from spacetrack import SpaceTrackClient

id_st   = os.getenv("id_st")
pass_st = os.getenv("pass_st")
st      = SpaceTrackClient(identity=id_st, password=pass_st)

# EDC API
id_edc  = os.getenv("id_edc")
pass_edc= os.getenv("pass_edc")
url_edc = 'https://edc.dgfi.tum.de/api/v1/'

# Orekit

import orekit
orekit.initVM()

from orekit.pyhelpers import setup_orekit_curdir
orekit_data_dir = 'orekit-data.zip'
setup_orekit_curdir(orekit_data_dir)

from org.orekit.utils import Constants as orekit_constants # type: ignore
from org.orekit.frames import FramesFactory # type: ignore
from org.orekit.utils import IERSConventions # type: ignore

tod     = FramesFactory.getTOD(IERSConventions.IERS_2010, False) # Taking tidal effects into account when interpolating EOP parameters
gcrf    = FramesFactory.getGCRF()
itrf    = FramesFactory.getITRF(IERSConventions.IERS_2010, False)
# Selecting frames to use for OD
eci     = gcrf
ecef    = itrf

from org.orekit.models.earth import ReferenceEllipsoid # type: ignore
from org.orekit.bodies import CelestialBodyFactory # type: ignore
from org.orekit.time import AbsoluteDate, TimeScalesFactory # type: ignore

wgs84Ellipsoid  = ReferenceEllipsoid.getWgs84(ecef)
moon            = CelestialBodyFactory.getMoon()
sun             = CelestialBodyFactory.getSun()
utc             = TimeScalesFactory.getUTC()
mjd_utc_epoch   = AbsoluteDate(1858, 11, 17, 0, 0, 0.0, utc)

# 

stationFile = 'SLRF2020_POS+VEL_2023.10.02.snx'
stationEccFile = 'ecc_xyz.snx'
from org.orekit.files.sinex import SinexLoader, Station # type: ignore
from org.orekit.data import DataSource # type: ignore
stations_map = SinexLoader(DataSource(stationFile)).getStations()
ecc_map = SinexLoader(DataSource(stationEccFile)).getStations()

from org.orekit.estimation.measurements import GroundStation # type: ignore
from org.orekit.frames import TopocentricFrame # type: ignore

station_keys = stations_map.keySet()

n_errors    = 0
station_df  = pd.DataFrame(columns=['lat_deg', 'lon_deg', 'alt_m', 'GroundStation'])
for key in station_keys:
    station_data    = stations_map.get(key)
    ecc_data        = ecc_map.get(key)
    if ecc_data.getEccRefSystem() != Station.ReferenceSystem.XYZ:
        print('Error, eccentricity coordinate system not XYZ')

    epoch_velocity = station_data.getEpoch()
    durationSinceEpoch = odDate.durationFrom(epoch_velocity)  # seconds

    # Computing current station position using velocity data
    station_pos_at_epoch = station_data.position
    vel = station_data.getVelocity()  # m/s
    station_pos_current = station_pos_at_epoch.add(vel.scalarMultiply(durationSinceEpoch))

    # Adding eccentricity
    try:
        station_pos_current = station_pos_current.add(ecc_data.getEccentricities(odDate))
        # Converting to ground station object
        geodeticPoint = wgs84Ellipsoid.transform(station_pos_current, itrf, odDate)
        lon_deg = np.rad2deg(geodeticPoint.getLongitude())
        lat_deg = np.rad2deg(geodeticPoint.getLatitude())
        alt_m   = geodeticPoint.getAltitude()
        topocentricFrame = TopocentricFrame(wgs84Ellipsoid, geodeticPoint, key)
        groundStation = GroundStation(topocentricFrame)
        station_df.loc[key] = [lat_deg, lon_deg, alt_m, groundStation]
    except:
        # And exception is thrown when the odDate is not in the date range of the eccentricity entry for this station
        # This is simply for stations which do not exist anymore at odDate
        n_errors += 1

station_df = station_df.sort_index()

# TLE data
from org.orekit.propagation.analytical.tle import TLE # type: ignore

tle = st.tle(norad_cat_id=sat_list[sc_name]['norad_id'], epoch='<{}'.format(absolutedate_to_datetime(odDate)), orderby='epoch desc', limit=1, format='tle').split('\n')
sat_list[sc_name]['TLE'] = TLE(tle[0],tle[1])

from org.orekit.attitudes import NadirPointing # type: ignore
from org.orekit.propagation.analytical.tle import SGP4 # type: ignore
from org.orekit.orbits import CartesianOrbit, PositionAngleType # type: ignore
from org.orekit.propagation.conversion import DormandPrince853IntegratorBuilder, NumericalPropagatorBuilder # type: ignore

nadirPointing = NadirPointing(eci, wgs84Ellipsoid)
sgp4Propagator = SGP4(sat_list[sc_name]['TLE'], nadirPointing, sat_list[sc_name]['mass'])

tleInitialState = sgp4Propagator.getInitialState()
tleEpoch        = tleInitialState.getDate()
tleOrbit_TEME   = tleInitialState.getOrbit()
tlePV_ECI       = tleOrbit_TEME.getPVCoordinates(eci)
tleOrbit_ECI    = CartesianOrbit(tlePV_ECI, eci, wgs84Ellipsoid.getGM())

integratorBuilder   = DormandPrince853IntegratorBuilder(prop_min_step, prop_max_step, prop_position_error)
propagatorBuilder   = NumericalPropagatorBuilder(tleOrbit_ECI, integratorBuilder, PositionAngleType.MEAN, estimator_position_scale)
propagatorBuilder.setMass(sat_list[sc_name]['mass'])
propagatorBuilder.setAttitudeProvider(nadirPointing)

# Perturbations and Forces

from org.orekit.forces.gravity.potential import GravityFieldFactory # type: ignore
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction, Relativity # type: ignore
from org.orekit.forces.radiation import IsotropicRadiationSingleCoefficient, SolarRadiationPressure # type: ignore
from org.orekit.models.earth.atmosphere.data import CssiSpaceWeatherData # type: ignore
from org.orekit.models.earth.atmosphere import NRLMSISE00 # type: ignore
from org.orekit.forces.drag import IsotropicDrag, DragForce # type: ignore

# Earth gravity field with degree 64 and order 64
gravityProvider = GravityFieldFactory.getNormalizedProvider(64, 64)
gravityAttractionModel = HolmesFeatherstoneAttractionModel(ecef, gravityProvider)
propagatorBuilder.addForceModel(gravityAttractionModel)

# Moon and Sun perturbations
moon_3dbodyattraction = ThirdBodyAttraction(moon)
propagatorBuilder.addForceModel(moon_3dbodyattraction)
sun_3dbodyattraction = ThirdBodyAttraction(sun)
propagatorBuilder.addForceModel(sun_3dbodyattraction)

# Solar radiation pressure
isotropicRadiationSingleCoeff = IsotropicRadiationSingleCoefficient(sat_list[sc_name]['cross_section'], sat_list[sc_name]['cr']);
solarRadiationPressure = SolarRadiationPressure(sun, wgs84Ellipsoid,
                                                isotropicRadiationSingleCoeff)
propagatorBuilder.addForceModel(solarRadiationPressure)

# Relativity
relativity = Relativity(orekit_constants.EIGEN5C_EARTH_MU)
propagatorBuilder.addForceModel(relativity)

# Atmospheric drag
cswl = CssiSpaceWeatherData("SpaceWeather-All-v1.2.txt")
atmosphere = NRLMSISE00(cswl, sun, wgs84Ellipsoid)
#from org.orekit.forces.drag.atmosphere import DTM2000
#atmosphere = DTM2000(msafe, sun, wgs84Ellipsoid)

isotropicDrag = IsotropicDrag(sat_list[sc_name]['cross_section'], sat_list[sc_name]['cd'])
dragForce = DragForce(atmosphere, isotropicDrag)
propagatorBuilder.addForceModel(dragForce)

# Set up estimator

from org.hipparchus.linear import QRDecomposer # type: ignore
from org.hipparchus.optim.nonlinear.vector.leastsquares import GaussNewtonOptimizer # type: ignore
from org.orekit.estimation.leastsquares import BatchLSEstimator # type: ignore

matrixDecomposer = QRDecomposer(1e-11)
optimizer = GaussNewtonOptimizer(matrixDecomposer, False)

estimator = BatchLSEstimator(optimizer, propagatorBuilder)
estimator.setParametersConvergenceThreshold(estimator_convergence_thres)
estimator.setMaxIterations(estimator_max_iterations)
estimator.setMaxEvaluations(estimator_max_evaluations)

                                        #######################
                                        # FETCHING RANGE DATA #
                                        #######################

from slrDataUtils import SlrDlManager
slr = SlrDlManager(username_edc=id_edc, password_edc=pass_edc, data_type=laser_data_type)
laserDatasetList = slr.querySlrData(laser_data_type, sat_list[sc_name]["cospar_id"],
                                    absolutedate_to_datetime(startCollectionDate), absolutedate_to_datetime(odDate))

slrDataFrame = slr.dlAndParseSlrData(laser_data_type, laserDatasetList, "slr_dataset")

from org.orekit.estimation.measurements import Range, ObservableSatellite # type: ignore
from org.orekit.models.earth.troposphere import MendesPavlisModel # type: ignore
from org.orekit.estimation.measurements.modifiers import RangeTroposphericDelayModifier # type: ignore
from org.orekit.models.earth.weather import GlobalPressureTemperature3 # type: ignore
from org.orekit.utils.units import Unit # type: ignore

observableSatellite = ObservableSatellite(0) # Propagator index = 0

observations = sum([int(obs) for obs in laserDatasetList.loc[:, "observations"]])

i = 1
while observations / i > 1000:
    i += 1
j = 0

for receiveTime, slrData in slrDataFrame.iterrows():
    if (station_id := slrData['station-id']) in station_df.index and not(j%i): # Checking if station exists in the stations list, because it might not be up-to-date
        if not np.isnan(slrData['range']):  # If this data point contains a valid range measurement
            orekitRange = Range(station_df.loc[station_id, 'GroundStation'],
                                True, # Two-way measurement
                                receiveTime,
                                slrData['range'],
                                range_sigma,
                                range_weight,
                                observableSatellite
                               ) # Uses date of signal reception; https://www.orekit.org/static/apidocs/org/orekit/estimation/measurements/Range.html

            range_tropo_delay_modifier = RangeTroposphericDelayModifier(MendesPavlisModel(GlobalPressureTemperature3(DataSource("gpt3_5.grd"), utc),
                                                                                          slrData["wavelength_microm"],
                                                                                          Unit.parse("µm")
                                                                                          )
                                                                        )
            orekitRange.addModifier(range_tropo_delay_modifier)
            estimator.addMeasurement(orekitRange)
    j += 1
    print(f"{100*j/observations:02.2f}%")
estimatedPropagatorArray = estimator.estimate()

# Propagating estimated orbit

dt = 300.0
date_start = startCollectionDate
date_start = date_start.shiftedBy(-86400.0)
date_end = odDate.shiftedBy(86400.0) # Stopping 1 day after OD date

# First propagating in ephemeris mode
estimatedPropagator = estimatedPropagatorArray[0]
estimatedInitialState = estimatedPropagator.getInitialState()
actualOdDate = estimatedInitialState.getDate()
estimatedPropagator.resetInitialState(estimatedInitialState)
eph_generator = estimatedPropagator.getEphemerisGenerator()

# Propagating from 1 day before data collection
# To 1 week after orbit determination (for CPF generation)
estimatedPropagator.propagate(date_start, odDate.shiftedBy(7 * 86400.0))
bounded_propagator = eph_generator.getGeneratedEphemeris()

# Covariance analysis

# Creating the LVLH frame
# It must be associated to the bounded propagator, not the original numerical propagator
from org.orekit.frames import LocalOrbitalFrame, LOFType # type: ignore
lvlh = LocalOrbitalFrame(eci, LOFType.LVLH, bounded_propagator, 'LVLH')

# Getting covariance matrix in ECI frame
covMat_eci_java = estimator.getPhysicalCovariances(1.0e-10)

# Converting matrix to LVLH frame
# Getting an inertial frame aligned with the LVLH frame at this instant
# The LVLH is normally not inertial, but this should not affect results too much
# Reference: David Vallado, Covariance Transformations for Satellite Flight Dynamics Operations, 2003
eci2lvlh_frozen = eci.getTransformTo(lvlh, actualOdDate).freeze()

# Computing Jacobian
from org.orekit.utils import CartesianDerivativesFilter
from orekit.pyhelpers import JArray_double2D
from org.hipparchus.linear import Array2DRowRealMatrix

jacobianDoubleArray = JArray_double2D(6, 6)
eci2lvlh_frozen.getJacobian(CartesianDerivativesFilter.USE_PV, jacobianDoubleArray)
jacobian = Array2DRowRealMatrix(jacobianDoubleArray)
# Applying Jacobian to convert matrix to lvlh
covMat_lvlh_java = jacobian.multiply(covMat_eci_java.multiply(jacobian.transpose()))

# Converting the Java matrices to numpy
covarianceMat_eci   = np.matrix([covMat_eci_java.getRow(iRow) for iRow in range(0, covMat_eci_java.getRowDimension())])
covarianceMat_lvlh  = np.matrix([covMat_lvlh_java.getRow(iRow) for iRow in range(0, covMat_lvlh_java.getRowDimension())])

pos_std_crossTrack = np.sqrt(covarianceMat_lvlh[0,0])
pos_std_alongTrack = np.sqrt(covarianceMat_lvlh[1,1])
pos_std_outOfPlane = np.sqrt(covarianceMat_lvlh[2,2])
print(f'Position std: cross-track {pos_std_crossTrack:.3e} m, along-track {pos_std_alongTrack:.3e} m, out-of-plane {pos_std_outOfPlane:.3e} m')

vel_std_crossTrack = np.sqrt(covarianceMat_lvlh[3,3])
vel_std_alongTrack = np.sqrt(covarianceMat_lvlh[4,4])
vel_std_outOfPlane = np.sqrt(covarianceMat_lvlh[5,5])
print(f'Velocity std: cross-track {vel_std_crossTrack:.3e} m/s, along-track {vel_std_alongTrack:.3e} m/s, out-of-plane {vel_std_outOfPlane:.3e} m/s')