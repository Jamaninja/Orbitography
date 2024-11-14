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
    sat_data['TLE'] = sat_data[['TLE_LINE1', 'TLE_LINE2']].apply(lambda tle: (tle.iloc[0], tle.iloc[1]), axis=1) # Combines most recent TLE values from TLE_LINE1 and TLE_LINE2 
    sat_data.drop(['TLE_LINE1', 'TLE_LINE2'], axis=1, inplace=True)                                              # into a tuple in a new column TLE

    return sat_data

def propagateOrbits(sat_data, resolution, duration):
    '''
        Propagates the satellite's orbit using SGP4/SDP4 propagation

        Args:
            sat_data: DataFrame
                Pandas DataFrame of satellites to propagate 
            
            resolution: float
                How many seconds between PV coordinate updates

            duration: int or float
                How many days to propagate orbits for
    '''

    sf                      = SatelliteFunctions(sat_data=sat_data)
    prop_data               = pd.DataFrame(columns=['OBJECT_NAME', 'OBJECT_TYPE', 'PVS', 'POS', 'VEL', 'GPS', 
                                                    'POS_x', 'POS_y', 'POS_z', 'VEL_x', 'VEL_y', 'VEL_z', 
                                                    'RADIUS', 'LATITUDE', 'LONGITUDE', 'EPOCH'], index=sat_data.index)
    prop_data[['OBJECT_NAME', 'OBJECT_TYPE']] = sat_data[['OBJECT_NAME', 'OBJECT_TYPE']]

    # Try to create empty directory to store temporary data in. If it exists, instead clear the directory
    try:
        os.makedirs('Temp_Propagation_Data')
    except:
        for file in os.listdir('Temp_Propagation_Data'):
            os.remove(f'Temp_Propagation_Data/{file}')

    batch_size = 2500
    batch_count = int(np.ceil(len(prop_data)/batch_size))

    # Process in batches
    for batch in range(batch_count):
        at_batch    = f'[{batch+1}/{batch_count}]'
        batch_data  = prop_data.iloc[batch*batch_size : (batch+1)*batch_size].copy()

        print(f'{at_batch} Propagating orbits...')
        batch_data['PVS']       = [sf.propagateTLE(sat=sat, resolution=resolution, duration=duration) for sat in batch_data.index]
        print(f'{at_batch} Extracting epochs...')
        batch_data['EPOCH']     = sat_data.iloc[batch*batch_size : (batch+1)*batch_size].TLE.apply(
            lambda tle: absolutedate_to_datetime(sf.toTLE(tle).getDate()).strftime('%Y/%m/%d %H:%M:%S.%f'))

        print(f'{at_batch} Calculating positions...')
        batch_data['POS']       = batch_data.PVS.apply(lambda pvs: list(map(lambda pv: pv.getPosition(), pvs)))
        print(f'{at_batch} ┠╴Extracting x coordinates...')
        batch_data['POS_x']     = batch_data.POS.apply(lambda pos: list(map(lambda p: p.x, pos))) # \
        print(f'{at_batch} ┠╴Extracting y coordinates...')                                        #  \
        batch_data['POS_y']     = batch_data.POS.apply(lambda pos: list(map(lambda p: p.y, pos))) # --> TODO: Condense into single statement
        print(f'{at_batch} ┖╴Extracting z coordinates...')                                        #  /
        batch_data['POS_z']     = batch_data.POS.apply(lambda pos: list(map(lambda p: p.z, pos))) # /

        # prop_data[['x', 'y', 'z']] = prop_data['PVS'].apply(lambda pvs: list(map(lambda pos: [pos.x, pos.y, pos.z], list(map(lambda pv: pv.getPosition(), pvs)))))

        print(f'{at_batch} Calculating radii...')
        batch_data['RADIUS']    = batch_data.POS.apply(lambda pos: list(map(lambda p: np.sqrt(p.x**2 + p.y**2 + p.z**2), pos)))

        print(f'{at_batch} Calculating velocities...')
        batch_data['VEL']       = batch_data.PVS.apply(lambda pvs: list(map(lambda pv: pv.getVelocity(), pvs)))
        print(f'{at_batch} ┠╴Extracting x components...')
        batch_data['VEL_x']     = batch_data.VEL.apply(lambda vel: list(map(lambda v: v.x, vel)))
        print(f'{at_batch} ┠╴Extracting y components...')
        batch_data['VEL_y']     = batch_data.VEL.apply(lambda vel: list(map(lambda v: v.y, vel)))
        print(f'{at_batch} ┖╴Extracting z components...')
        batch_data['VEL_z']     = batch_data.VEL.apply(lambda vel: list(map(lambda v: v.z, vel)))

        print(f'{at_batch} Calculating groundpoints...')
        batch_data['GPS']       = batch_data.PVS.apply(lambda pvs: list(map(lambda pv: sf.earth(sf.eme2000).transform(pv.position, sf.eme2000, pv.date), pvs)))
        print(f'{at_batch} ┠╴Extracting latitudes...')
        batch_data['LATITUDE']  = batch_data.GPS.apply(lambda gps: list(map(lambda gp: np.degrees(gp.latitude), gps)))
        print(f'{at_batch} ┖╴Extracting longitudes...')
        batch_data['LONGITUDE'] = batch_data.GPS.apply(lambda gps: list(map(lambda gp: np.degrees(gp.longitude), gps)))
        
        print(f'{at_batch} Saving batch...\n')
        batch_data.filter(['OBJECT_NAME', 'OBJECT_TYPE',
                          'POS_x', 'POS_y', 'POS_z', 'VEL_x', 'VEL_y', 'VEL_z',
                          'RADIUS', 'LATITUDE', 'LONGITUDE', 'EPOCH']).to_json(f'Temp_Propagation_Data/temp_propagation_data_{batch:03}.json')

    print('Combining batches...')
    files = os.listdir('Temp_Propagation_Data')
    for batch_file in files:
        with open(f'Temp_Propagation_Data/{batch_file}') as file:
            batch_data = pd.read_json(file)
        os.remove(f'Temp_Propagation_Data/{batch_file}')
        prop_data.update(batch_data)
    prop_data.filter(['OBJECT_NAME', 'OBJECT_TYPE',
                      'POS_x', 'POS_y', 'POS_z', 'VEL_x', 'VEL_y', 'VEL_z',
                      'RADIUS', 'LATITUDE', 'LONGITUDE', 'EPOCH']).to_json('Propagation_Data.json')
    
    print('Removing temporary files...')
    os.rmdir('Temp_Propagation_Data')

with open((metadata_file := 'Metadata.json'), 'r') as file:
    metadata = json.load(file)

propagateOrbits(importDatabase(metadata['path']['database']), resolution:=300., duration:=0.25)

# Update metadata.json
metadata['objects'] = {
    'PAYLOAD'       : True,
    'ROCKET BODY'   : False,
    'DEBRIS'        : False,
    'UNKNOWN'       : False,
    'OTHER'         : False
    }

metadata['datetimes'] = {
    'start':    str(now_UTC),
    'timestep': resolution,
    'steps':    int(duration * 86400 / resolution)
    }

with open(metadata_file, 'w') as file:
    json.dump(metadata, file)

# Plot propagation data if desired
if True:
    print('Tracing orbits...')
    pf = PlotFunctions(prop_data_file='Propagation_Data.json')
    pf.plotOrbits(metadata_file=metadata_file, limit=500)