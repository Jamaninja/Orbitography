import orekit
vm = orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir, absolutedate_to_datetime
setup_orekit_curdir()

import pandas as pd
import numpy as np
from Orbitography_Functions import SatelliteFunctions, PlotFunctions, now_UTC
import json

def importDatabase(database_path):
    sat_data        = pd.read_json(database_path).filter(['OBJECT_NAME', 'OBJECT_TYPE', 'TLE_LINE1', 'TLE_LINE2'])
    sat_data['TLE'] = sat_data[['TLE_LINE1', 'TLE_LINE2']].apply(lambda x: (x.iloc[0], x.iloc[1]), axis=1)  # Combines most recent TLE values from TLE_LINE1 and TLE_LINE2 
    sat_data        = sat_data.drop(['TLE_LINE1', 'TLE_LINE2'], axis=1)                                     # into a tuple in a new column TLE

    return sat_data

def propagateOrbits(sat_data, resolution, duration):
    sf                      = SatelliteFunctions(sat_data=sat_data)
    prop_data               = pd.DataFrame(columns=['object_name', 'object_type', 'pvs', 'pos', 'vel', 'gps', 
                                                    'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 
                                                    'radius', 'latitude', 'longitude', 'epoch'], index=sat_data.index)
    prop_data[['object_name', 'object_type']] = sat_data[['OBJECT_NAME', 'OBJECT_TYPE']]

    print('Propagating orbits...')
    prop_data['pvs']        = [sf.propagateTLE(sat=sat, resolution=resolution, duration=duration) for sat in sat_data.index]
    print('Extracting epochs...')
    prop_data['epoch']      = sat_data['TLE'].apply(lambda tle: absolutedate_to_datetime(sf.toTLE(tle).getDate()).strftime('%Y/%m/%d %H:%M:%S.%f'))

    print('Calculating positions...')
    prop_data['pos']        = prop_data['pvs'].apply(lambda pvs: list(map(lambda pv: pv.getPosition(), pvs)))
    print('    Calculating x coordinates...')
    prop_data['pos_x']      = prop_data['pos'].apply(lambda pos: list(map(lambda p: p.x, pos))) # \
    print('    Calculating y coordinates...')                                                       #  \
    prop_data['pos_y']      = prop_data['pos'].apply(lambda pos: list(map(lambda p: p.y, pos))) # --> TODO: Condense into single statement
    print('    Calculating z coordinates...')                                                       #  /
    prop_data['pos_z']      = prop_data['pos'].apply(lambda pos: list(map(lambda p: p.z, pos))) # /

    # prop_data[['x', 'y', 'z']] = prop_data['pvs'].apply(lambda pvs: list(map(lambda pos: [pos.x, pos.y, pos.z], list(map(lambda pv: pv.getPosition(), pvs)))))

    print('Calculating radii...')
    prop_data['radius']     = prop_data['pos'].apply(lambda pos: list(map(lambda p: np.sqrt(p.x**2 + p.y**2 + p.z**2), pos)))

    print('Calculating velocities...')
    prop_data['vel']        = prop_data['pvs'].apply(lambda pvs: list(map(lambda pv: pv.getVelocity(), pvs)))
    print('    Calculating x component...')
    prop_data['vel_x']      = prop_data['vel'].apply(lambda vel: list(map(lambda v: v.x, vel)))
    print('    Calculating y component...')
    prop_data['vel_y']      = prop_data['vel'].apply(lambda vel: list(map(lambda v: v.y, vel)))
    print('    Calculating z component...')
    prop_data['vel_z']      = prop_data['vel'].apply(lambda vel: list(map(lambda v: v.z, vel)))

    print('Calculating groundpoints...')
    prop_data['gps']        = prop_data['pvs'].apply(lambda pvs: list(map(lambda pv: sf.earth(sf.eme2000).transform(pv.position, sf.eme2000, pv.date), pvs)))
    print('    Calculating latitudes...')
    prop_data['latitude']   = prop_data['gps'].apply(lambda gps: list(map(lambda gp: np.degrees(gp.latitude), gps)))
    print('    Calculating longitudes...')
    prop_data['longitude']  = prop_data['gps'].apply(lambda gps: list(map(lambda gp: np.degrees(gp.longitude), gps)))

    prop_data.drop(['pvs', 'pos', 'vel', 'gps'], axis=1).to_json('Propagation_Data.json')

    dt_data = {
        'start':    str(now_UTC),
        'timestep': resolution,
        'steps':    int(duration * 86400 / resolution)
    }
    
    return dt_data

with open((metadata_file := 'Metadata.json'), 'r') as file:
    metadata = json.load(file)

sat_data    = importDatabase(metadata['path']['database'])
dt_data     = propagateOrbits(sat_data, 30., .25)

# Update metadata.json
metadata['objects'] = {
    'PAYLOAD'       : True,
    'ROCKET BODY'   : False,
    'DEBRIS'        : False,
    'UNKNOWN'       : False,
    'OTHER'         : False
    }

metadata['datetimes'] = dt_data

with open(metadata_file, 'w') as file:
    json.dump(metadata, file)

# Plot propagation data if desired
if False:
    print('Tracing orbits...')
    pf = PlotFunctions(prop_data_file='Propagation_Data.json')
    pf.plotOrbits(metadata_file=metadata_file, limit=500)