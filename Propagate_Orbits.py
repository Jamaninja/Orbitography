import orekit
vm = orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir, absolutedate_to_datetime
setup_orekit_curdir()

import pandas as pd
import numpy as np
from Orbitography_Functions import SatelliteFunctions, PlotFunctions
import json

def importDatabase(database_path):
    sat_data        = pd.read_json(database_path).filter(['OBJECT_NAME', 'OBJECT_TYPE', 'TLE_LINE1', 'TLE_LINE2'])
    sat_data['TLE'] = sat_data[['TLE_LINE1', 'TLE_LINE2']].apply(lambda x: (x.iloc[0], x.iloc[1]), axis=1)    # Combines most recent TLE values from TLE_LINE1 and TLE_LINE2 
    sat_data        = sat_data.drop(['TLE_LINE1', 'TLE_LINE2'], axis=1)                                             # into a tuple in a new column TLE

    return sat_data

def propagateOrbits(sat_data):
    sf                      = SatelliteFunctions(sat_data=sat_data)
    prop_data               = pd.DataFrame(columns=['object_name',  'object_type', 'pvs', 'pos', 'gps', 'x', 'y', 'z', 'radius', 'latitude', 'longitude', 'epoch'], index=sat_data.index)
    prop_data[['object_name', 'object_type']] = sat_data[['OBJECT_NAME', 'OBJECT_TYPE']]

    print('Propagating orbits...')
    prop_data['pvs']        = [sf.propagateTLE(sat=sat, resolution=300., duration=.25) for sat in sat_data.index]
    print('Extracting epochs...')
    prop_data.update(sat_data['TLE'].apply(lambda x: sf.toTLE(x).getDate()))

    print('Calculating positions...')
    prop_data['pos']        = prop_data['pvs'].apply(lambda pvs: list(map(lambda pv: pv.getPosition(), pvs)))
    print('Calculating x coordinates...')
    prop_data['x']          = prop_data['pos'].apply(lambda pos: list(map(lambda p: p.x, pos))) # \
    print('Calculating y coordinates...')                                                       #  \
    prop_data['y']          = prop_data['pos'].apply(lambda pos: list(map(lambda p: p.y, pos))) # --> TODO: Condense into single statement
    print('Calculating z coordinates...')                                                       #  /
    prop_data['z']          = prop_data['pos'].apply(lambda pos: list(map(lambda p: p.z, pos))) # /

    # prop_data[['x', 'y', 'z']] = prop_data['pvs'].apply(lambda pvs: list(map(lambda pos: [pos.x, pos.y, pos.z], list(map(lambda pv: pv.getPosition(), pvs)))))

    print('Calculating radii...')
    prop_data['radius']     = prop_data['pos'].apply(lambda pos: list(map(lambda p: np.sqrt(p.x**2 + p.y**2 + p.z**2), pos)))

    print('Calculating groundpoints...')
    prop_data['gps']        = prop_data['pvs'].apply(lambda pvs: list(map(lambda pv: sf.earth(sf.eme2000).transform(pv.position, sf.eme2000, pv.date), pvs)))
    print('Calculating latitudes...')
    prop_data['latitude']   = prop_data['gps'].apply(lambda gps: list(map(lambda gp: np.degrees(gp.latitude), gps)))
    print('Calculating longitudes...')
    prop_data['longitude']  = prop_data['gps'].apply(lambda gps: list(map(lambda gp: np.degrees(gp.longitude), gps)))

    prop_data.drop(['pvs', 'pos', 'gps'], axis=1).to_json('Propagation_Data.json')
    return prop_data

def updateMetadata(prop_data):
    datetimes   = [absolutedate_to_datetime(pv.getDate()) for pv in prop_data.iloc[0]['pvs']]
    objects     = {
    'PAYLOAD'       : True,
    'ROCKET BODY'   : False,
    'DEBRIS'        : False,
    'UNKNOWN'       : False,
    'OTHER'         : False
    }

    with open((metadata_file := 'Metadata.json'), 'r') as file:
        metadata = json.load(file)

    metadata['objects']     = objects
    metadata['datetimes']   = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in datetimes]

    with open(metadata_file, 'w') as file:
        json.dump(metadata, file)

with open((metadata_file := 'Metadata.json'), 'r') as file:
    metadata = json.load(file)

sat_data    = importDatabase(metadata['path']['database'])
prop_data   = propagateOrbits(sat_data)
updateMetadata(prop_data)

print('Tracing orbits...')
pf = PlotFunctions(prop_data_file='Propagation_Data.json')
pf.plotOrbits(metadata_file='Metadata.json')