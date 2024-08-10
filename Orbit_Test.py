import orekit
vm = orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir, absolutedate_to_datetime
setup_orekit_curdir()

import pandas as pd
import numpy as np
from Orbitography_Functions import SatelliteFunctions, PlotFunctions

sat_data = pd.read_pickle("Satellite_Data.pkl")
sf = SatelliteFunctions(sat_data=sat_data)

prop_data = {}

for sat in sat_data:
    pvs                                 = sf.propagateNumerical(sat=sat, resolution=60.0, duration=.25)
    groundpoints                        = [sf.earth(sf.eme2000).transform(pv.position, sf.eme2000, pv.date) for pv in pvs]
    prop_data.update(     {sat          : {}})
    prop_data[sat].update({"x"          : [pv.getPosition().x for pv in pvs],
                           "y"          : [pv.getPosition().y for pv in pvs],
                           "z"          : [pv.getPosition().z for pv in pvs]})   
    prop_data[sat].update({"radius"     : [np.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(prop_data[sat]["x"], prop_data[sat]["y"], prop_data[sat]["z"])],
                           "datetime"   : [absolutedate_to_datetime(pv.getDate()) for pv in pvs],
                           "latitude"   : np.degrees([gp.latitude for gp in groundpoints]),
                           "longitude"  : np.degrees([gp.longitude for gp in groundpoints]),
                           "epoch"      : absolutedate_to_datetime(sf.toTLE(sat_data.get(sat)["TLE"]).getDate())})

pd.DataFrame(prop_data).to_pickle("Propagation_Data.pkl")

pf = PlotFunctions(prop_data_file="Propagation_Data.pkl")
pf.plotOrbits()