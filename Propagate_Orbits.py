import orekit
vm = orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir, absolutedate_to_datetime
setup_orekit_curdir()

import pandas as pd
import numpy as np
import os
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

    os.makedirs('Temp_Propagation_Data')
    batch_size = 1000
    batch_count = int(np.ceil(len(prop_data)/batch_size))

    # Process in batches
    for batch in range(batch_count):
        at_batch    = f'[{batch+1}/{batch_count}]'
        batch_data  = prop_data.iloc[batch*batch_size : (batch+1)*batch_size]

        print(f'{at_batch} Propagating orbits...')
        batch_data['pvs']       = [sf.propagateTLE(sat=sat, resolution=resolution, duration=duration) for sat in batch_data.index]
        print(f'{at_batch} Extracting epochs...')
        batch_data['epoch']     = sat_data.iloc[batch*batch_size : (batch+1)*batch_size]['TLE'].apply(lambda tle: absolutedate_to_datetime(sf.toTLE(tle).getDate()).strftime('%Y/%m/%d %H:%M:%S.%f'))

        print(f'{at_batch} Calculating positions...')
        batch_data['pos']       = batch_data['pvs'].apply(lambda pvs: list(map(lambda pv: pv.getPosition(), pvs)))
        print(f'{at_batch} ┠╴Extracting x coordinates...')
        batch_data['pos_x']     = batch_data['pos'].apply(lambda pos: list(map(lambda p: p.x, pos))) # \
        print(f'{at_batch} ┠╴Extracting y coordinates...')                                        #  \
        batch_data['pos_y']     = batch_data['pos'].apply(lambda pos: list(map(lambda p: p.y, pos))) # --> TODO: Condense into single statement
        print(f'{at_batch} ┖╴Extracting z coordinates...')                                        #  /
        batch_data['pos_z']     = batch_data['pos'].apply(lambda pos: list(map(lambda p: p.z, pos))) # /

        # prop_data[['x', 'y', 'z']] = prop_data['pvs'].apply(lambda pvs: list(map(lambda pos: [pos.x, pos.y, pos.z], list(map(lambda pv: pv.getPosition(), pvs)))))

        print(f'{at_batch} Calculating radii...')
        batch_data['radius']    = batch_data['pos'].apply(lambda pos: list(map(lambda p: np.sqrt(p.x**2 + p.y**2 + p.z**2), pos)))

        print(f'{at_batch} Calculating velocities...')
        batch_data['vel']       = batch_data['pvs'].apply(lambda pvs: list(map(lambda pv: pv.getVelocity(), pvs)))
        print(f'{at_batch} ┠╴Extracting x components...')
        batch_data['vel_x']     = batch_data['vel'].apply(lambda vel: list(map(lambda v: v.x, vel)))
        print(f'{at_batch} ┠╴Extracting y components...')
        batch_data['vel_y']     = batch_data['vel'].apply(lambda vel: list(map(lambda v: v.y, vel)))
        print(f'{at_batch} ┖╴Extracting z components...')
        batch_data['vel_z']     = batch_data['vel'].apply(lambda vel: list(map(lambda v: v.z, vel)))

        print(f'{at_batch} Calculating groundpoints...')
        batch_data['gps']       = batch_data['pvs'].apply(lambda pvs: list(map(lambda pv: sf.earth(sf.eme2000).transform(pv.position, sf.eme2000, pv.date), pvs)))
        print(f'{at_batch} ┠╴Extracting latitudes...')
        batch_data['latitude']  = batch_data['gps'].apply(lambda gps: list(map(lambda gp: np.degrees(gp.latitude), gps)))
        print(f'{at_batch} ┖╴Extracting longitudes...')
        batch_data['longitude'] = batch_data['gps'].apply(lambda gps: list(map(lambda gp: np.degrees(gp.longitude), gps)))
        
        print(f'{at_batch} Saving batch...\n')
        batch_data.filter(['object_name', 'object_type',
                          'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z',
                          'radius', 'latitude', 'longitude', 'epoch']).to_json(f'Temp_Propagation_Data/temp_propagation_data_{batch:03}.json')

    print('Combining batches...')
    files = os.listdir('Temp_Propagation_Data')
    for batch_file in files:
        with open(f'Temp_Propagation_Data/{batch_file}') as file:
            batch_data = pd.read_json(file)
        os.remove(f'Temp_Propagation_Data/{batch_file}')
        prop_data.update(batch_data)
    prop_data.filter(['object_name', 'object_type',
                      'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z',
                      'radius', 'latitude', 'longitude', 'epoch']).to_json('Propagation_Data.json')
    
    print('Removing temporary files...')
    os.rmdir('Temp_Propagation_Data')

    dt_data = {
        'start':    str(now_UTC),
        'timestep': resolution,
        'steps':    int(duration * 86400 / resolution)
        }
    
    return dt_data

with open((metadata_file := 'Metadata.json'), 'r') as file:
    metadata = json.load(file)

sat_data    = importDatabase(metadata['path']['database'])
dt_data     = propagateOrbits(sat_data, 30., .1)

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
if True:
    print('Tracing orbits...')
    pf = PlotFunctions(prop_data_file='Propagation_Data.json')
    pf.plotOrbits(metadata_file=metadata_file, limit=500)