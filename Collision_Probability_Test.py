import orekit
vm = orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir
setup_orekit_curdir()

from org.orekit.data import DataProvidersManager, ZipJarCrawler # type: ignore
from org.orekit.frames import FramesFactory, TopocentricFrame # type: ignore
from org.orekit.bodies import OneAxisEllipsoid, GeodeticPoint # type: ignore
from org.orekit.time import TimeScalesFactory, AbsoluteDate, DateTimeComponents # type: ignore
from org.orekit.utils import IERSConventions, Constants, PVCoordinates # type: ignore
from org.orekit.orbits import CartesianOrbit, OrbitType, PositionAngleType, KeplerianOrbit, CircularOrbit, EquinoctialOrbit # type: ignore
from org.orekit.propagation.analytical.tle import TLE, TLEPropagator # type: ignore
from org.orekit.propagation import StateCovariance # type: ignore
from org.orekit.ssa.collision.shorttermencounter.probability.twod import Patera2005 # type: ignore
from org.hipparchus.geometry.euclidean.threed import Vector3D # type: ignore
from org.hipparchus.linear import MatrixUtils # type: ignore
from org.orekit.ssa.metrics import ProbabilityOfCollision # type: ignore
from java.io import File # type: ignore

from random import random


state1 = [700001.0, 0.0, 0.0, 0.0, 7000.24, 0.0]
state2 = [700000.0, 0.0, 0.0, 0.0, 7000.10, 0.0]
tca = AbsoluteDate(2020, 1, 1, 0, 0, 0.0, TimeScalesFactory.getUTC())

itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, False)
eme = FramesFactory.getEME2000()

orbit1 = CartesianOrbit(PVCoordinates(Vector3D(float(state1[0]), float(state1[1]), float(state1[2])),
                        Vector3D(float(state1[3]), float(state1[4]), float(state1[5]))),
                        eme,
                        tca,
                        Constants.WGS84_EARTH_MU)

orbit2 = CartesianOrbit(PVCoordinates(Vector3D(float(state2[0]), float(state2[1]), float(state2[2])),
                        Vector3D(float(state2[3]), float(state2[4]), float(state2[5]))),
                        eme,
                        tca,
                        Constants.WGS84_EARTH_MU)

radius1 = 1.0
radius2 = 1.0

cov_mat1 = MatrixUtils.createRealMatrix(6,6)
cov_mat2 = MatrixUtils.createRealMatrix(6,6)

for i in range(6):
    cov = random()
    cov_mat1.setEntry(i, i, cov)

    cov = random()
    cov_mat2.setEntry(i, i, cov)

    for j in range(i+1, 6):
        cov = random()
        cov_mat1.setEntry(i, j, cov)
        cov_mat1.setEntry(j, i, cov)

        cov = random()
        cov_mat2.setEntry(i, j, cov)
        cov_mat2.setEntry(j, i, cov)

covariance1 = StateCovariance(cov_mat1, tca, itrf, OrbitType.CARTESIAN, PositionAngleType.TRUE)
covariance2 = StateCovariance(cov_mat2, tca, itrf, OrbitType.CARTESIAN, PositionAngleType.TRUE)

try:
    poc_result = Patera2005().compute(orbit1, covariance1, orbit2, covariance2, radius1 + radius2, 1e-10)
    print("Probability of collision: {:.4f}%".format(100*poc_result.getValue()))
except:
    print('Probability of collision: 0.00%')