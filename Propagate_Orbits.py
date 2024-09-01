import orekit
vm = orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir, absolutedate_to_datetime
setup_orekit_curdir()

import pandas as pd
import numpy as np
from Orbitography_Functions import SatelliteFunctions, PlotFunctions

sat_data        = pd.read_pickle('Ephemeris_Database/Database.pkl').filter(['NORAD_CAT_ID', 'OBJECT_NAME', 'TLE_LINE1', 'TLE_LINE2']).set_index('NORAD_CAT_ID')
sat_data['TLE'] = sat_data[['TLE_LINE1', 'TLE_LINE2']].apply(lambda x: (x.iloc[0][0], x.iloc[1][0]), axis=1) # Combines columns TLE_LINE1 and TLE_LINE2 into a tuple in a new column TLE 
sat_data        = sat_data.drop(['TLE_LINE1', 'TLE_LINE2'], axis=1)

sf          = SatelliteFunctions(sat_data=sat_data)
prop_data   = pd.DataFrame(columns=['pvs', 'pos', 'gps', 'x', 'y', 'z', 'radius', 'datetime', 'latitude', 'longitude', 'epoch'], index = sat_data.index)
n           = 0
length      = len(sat_data)

print('Propagating orbits...')
for sat in sat_data.index:
    prop_data.loc[sat, 'pvs']   = sf.propagateTLE(sat=sat, resolution=300., duration=.25)
    prop_data.loc[sat, 'epoch'] = absolutedate_to_datetime(sf.toTLE(sat_data.loc[sat, 'TLE']).getDate())
    n += 1
    if not n % 50:
        print(f'Propagating orbits:  {100*n/length:00.2f}%')
print('Propagating orbits: 100.00%')

print('Calculating positions...')
prop_data.loc[:, 'pos']         = prop_data.loc[:, 'pvs'].apply(lambda pvs: list(map(lambda pv: pv.getPosition(), pvs)))
print('Calculating x coordinates...')
prop_data.loc[:, 'x']           = prop_data.loc[:, 'pos'].apply(lambda pos: list(map(lambda p: p.x, pos))) # \
print('Calculating y coordinates...')                                                                      #  \
prop_data.loc[:, 'y']           = prop_data.loc[:, 'pos'].apply(lambda pos: list(map(lambda p: p.y, pos))) # --> TODO: Condense into single statement
print('Calculating z coordinates...')                                                                      #  /
prop_data.loc[:, 'z']           = prop_data.loc[:, 'pos'].apply(lambda pos: list(map(lambda p: p.z, pos))) # /
print('Calculating radii...')
prop_data.loc[:, 'radius']      = prop_data.loc[:, 'pos'].apply(lambda pos: np.sqrt(list(map(lambda p: p.x**2 + p.y**2 + p.z**2, pos))))
print('Calculating datetimes...')
prop_data.loc[:, 'datetime']    = prop_data.loc[:, 'pvs'].apply(lambda pvs: list(map(lambda pv: absolutedate_to_datetime(pv.getDate()), pvs)))
print('Calculating groundpoints...')
prop_data.loc[:, 'gps']         = prop_data.loc[:, 'pvs'].apply(lambda pvs: list(map(lambda pv: sf.earth(sf.eme2000).transform(pv.position, sf.eme2000, pv.date), pvs)))
print('Calculating latitudes...')
prop_data.loc[:, 'latitude']    = prop_data.loc[:, 'gps'].apply(lambda gps: np.degrees(list(map(lambda gp: gp.latitude, gps))))
print('Calculating longitudes...')
prop_data.loc[:, 'longitude']   = prop_data.loc[:, 'gps'].apply(lambda gps: np.degrees(list(map(lambda gp: gp.longitude, gps))))

print('Saving propagation data...')
prop_data = prop_data.drop(['pvs', 'pos', 'gps'], axis=1)
prop_data.to_pickle('Propagation_Data_2.pkl')

print('Tracing orbits...')
pf = PlotFunctions(prop_data_file='Propagation_Data_2.pkl')
pf.plotOrbits()