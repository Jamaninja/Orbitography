import orekit
vm = orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir, absolutedate_to_datetime
setup_orekit_curdir()

import pandas as pd
import numpy as np
from Orbitography_Functions import SatelliteFunctions, PlotFunctions
import json

def importDatabase(database_path):
    sat_data        = pd.read_pickle(database_path).filter(['NORAD_CAT_ID', 'OBJECT_NAME', 'OBJECT_TYPE', 'TLE_LINE1', 'TLE_LINE2']).set_index('NORAD_CAT_ID')
    sat_data['TLE'] = sat_data[['TLE_LINE1', 'TLE_LINE2']].apply(lambda x: (x.iloc[0][0], x.iloc[1][0]), axis=1) # Combines TLE_LINE1 and TLE_LINE2 into a tuple in a new column TLE 
    sat_data        = sat_data.drop(['TLE_LINE1', 'TLE_LINE2'], axis=1)

    return sat_data

def propagateOrbits(sat_data):
    sf                              = SatelliteFunctions(sat_data=sat_data)
    prop_data                       = pd.DataFrame(columns=['object_name', 'pvs', 'pos', 'gps', 'x', 'y', 'z', 'radius', 'latitude', 'longitude', 'epoch', 'object_type'], index = sat_data.index)
    prop_data.loc[:, 'object_name'] = sat_data.loc[:, 'OBJECT_NAME']
    prop_data.loc[:, 'object_type'] = sat_data.loc[:, 'OBJECT_TYPE']

    print('Propagating orbits...')
    prop_data.loc[:, 'pvs']         = [sf.propagateTLE(sat=sat, resolution=300., duration=.25) for sat in sat_data.index]
    print('Extracting epochs...')
    prop_data.loc[:, 'epoch']       = [absolutedate_to_datetime(sf.toTLE(sat_data.loc[sat, 'TLE']).getDate()) for sat in sat_data.index]

    print('Calculating positions...')
    prop_data.loc[:, 'pos']         = prop_data.loc[:, 'pvs'].apply(lambda pvs: list(map(lambda pv: pv.getPosition(), pvs)))
    print('Calculating x coordinates...')
    prop_data.loc[:, 'x']           = prop_data.loc[:, 'pos'].apply(lambda pos: list(map(lambda p: p.x, pos))) # \
    print('Calculating y coordinates...')                                                                      #  \
    prop_data.loc[:, 'y']           = prop_data.loc[:, 'pos'].apply(lambda pos: list(map(lambda p: p.y, pos))) # --> TODO: Condense into single statement
    print('Calculating z coordinates...')                                                                      #  /
    prop_data.loc[:, 'z']           = prop_data.loc[:, 'pos'].apply(lambda pos: list(map(lambda p: p.z, pos))) # /
    print('Calculating radii...')
    prop_data.loc[:, 'radius']      = prop_data.loc[:, 'pos'].apply(lambda pos: list(map(lambda p: np.sqrt(p.x**2 + p.y**2 + p.z**2), pos)))

    print('Calculating groundpoints...')
    prop_data.loc[:, 'gps']         = prop_data.loc[:, 'pvs'].apply(lambda pvs: list(map(lambda pv: sf.earth(sf.eme2000).transform(pv.position, sf.eme2000, pv.date), pvs)))
    print('Calculating latitudes...')
    prop_data.loc[:, 'latitude']    = prop_data.loc[:, 'gps'].apply(lambda gps: list(map(lambda gp: np.degrees(gp.latitude), gps)))
    print('Calculating longitudes...')
    prop_data.loc[:, 'longitude']   = prop_data.loc[:, 'gps'].apply(lambda gps: list(map(lambda gp: np.degrees(gp.longitude), gps)))

    prop_data.drop(['pvs', 'pos', 'gps'], axis=1).to_pickle('Propagation_Data.pkl')
    return prop_data

def configureMetadata(prop_data):
    datetimes   = [absolutedate_to_datetime(pv.getDate()) for pv in prop_data.iloc[0]['pvs']]
    objects     = {
    'PAYLOAD'       : True,
    'ROCKET BODY'   : False,
    'DEBRIS'        : False,
    'UNKNOWN'       : False,
    'OTHER'         : False
    }

    prop_metadata = {
        'datetimes' : [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in datetimes],
        'objects'   : objects
        }
    
    with open('Propagation_Metadata.json', 'w') as file:
        json.dump(prop_metadata, file)

sat_data    = importDatabase('Satellite_Database/Database.json')
prop_data   = propagateOrbits(sat_data)
configureMetadata(prop_data)

print('Tracing orbits...')
pf = PlotFunctions(prop_data_file='Propagation_Data.pkl')
pf.plotOrbits(prop_metadata_file='Propagation_Metadata.json')