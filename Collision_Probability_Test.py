import orekit
vm = orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir, absolutedate_to_datetime
setup_orekit_curdir()

from org.orekit.data import DataProvidersManager, ZipJarCrawler # type: ignore
from org.orekit.frames import FramesFactory, TopocentricFrame # type: ignore
from org.orekit.bodies import OneAxisEllipsoid, GeodeticPoint # type: ignore
from org.orekit.time import TimeScalesFactory, AbsoluteDate, DateTimeComponents # type: ignore
from org.orekit.utils import IERSConventions, Constants, PVCoordinates # type: ignore
from org.orekit.orbits import CartesianOrbit, OrbitType, PositionAngleType # type: ignore

from org.orekit.propagation.analytical.tle import TLE, TLEPropagator # type: ignore
from org.orekit.propagation import StateCovariance # type: ignore
from org.orekit.ssa.collision.shorttermencounter.probability.twod import Patera2005 # type: ignore
from org.hipparchus.geometry.euclidean.threed import Vector3D # type: ignore
from java.io import File # type: ignore

import datetime, random
import numpy as np

state1 = [7000000.0, 0.0, 0.0, 0.0, 7000.0, 0.0]
state2 = [7100000.0, 0.0, 0.0, 0.0, 7000.1, 0.0]
tca = AbsoluteDate(2020, 1, 1, 0, 0, 0.0, TimeScalesFactory.getUTC())

orbit1 = CartesianOrbit(PVCoordinates(Vector3D(state1[0], state1[1], state1[2]),
                        Vector3D(state1[3], state1[4], state1[5])),
                        FramesFactory.getEME2000(),
                        tca,
                        Constants.WGS84_EARTH_MU)

orbit2 = CartesianOrbit(PVCoordinates(Vector3D(state2[0], state2[1], state2[2]),
                        Vector3D(state2[3], state2[4], state2[5])),
                        FramesFactory.getEME2000(),
                        tca,
                        Constants.WGS84_EARTH_MU)

radius1 = 1.0
radius2 = 1.0

cov_mat1 = np.zeros(shape=(6,6))
cov_mat2 = np.zeros(shape=(6,6))

for i in range(6):
    for j in range(6):
        cov_mat1[i,j] = random.random()
        cov_mat2[i,j] = random.random()

covariance1 = StateCovariance(cov_mat1, tca, FramesFactory.getEME2000(), OrbitType.CARTESIAN, PositionAngleType.TRUE)
covariance2 = StateCovariance(cov_mat2, tca, FramesFactory.getEME2000(), OrbitType.CARTESIAN, PositionAngleType.TRUE)

# Patera2005.compute(Orbit primaryAtTCA, StateCovariance primaryCovariance, double primaryRadius, Orbit secondaryAtTCA, StateCovariance secondaryCovariance, double secondaryRadius)
poc_result = Patera2005().compute(orbit1, covariance1, radius1, orbit2, covariance2, radius2)
print(f"Probability of collision: {poc_result}")