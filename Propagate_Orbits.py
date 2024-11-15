import orekit
vm = orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir, absolutedate_to_datetime
setup_orekit_curdir()

import pandas as pd
import numpy as np
import os
from Orbitography_Functions import PlotOrbit, SatellitePropagation, now_UTC
import json

def runPropagation(prop_func, database, resolution, duration, **kwargs):

    batch_size = kwargs.get('batch_size')

    TLE_data    = sp.importTLE(database)
    prop_data   = TLE_data.copy().join(pd.DataFrame(columns=['EPOCH', 'PVS', 'POS', 'POS_x', 'POS_y', 'POS_z', 'RADIUS',
                                                             'VEL', 'VEL_x', 'VEL_y', 'VEL_z', 'GPS', 'LATITUDE', 'LONGITUDE']))

    if batch_size:

        # Try to create empty directory to store temporary data in. If it exists, instead clear the directory
        try:
            os.makedirs('Temp_Propagation_Data')
        except:
            for file in os.listdir('Temp_Propagation_Data'):
                os.remove(f'Temp_Propagation_Data/{file}')
        batch_count = int(np.ceil(len(prop_data)/batch_size))

        # Process in batches
        for batch_no in range(batch_count):
            batch_prog  = f'[{batch_no+1}/{batch_count}] '
            batch_data  = prop_data.iloc[batch_no*batch_size : (batch_no+1)*batch_size].copy()

            print(f'{batch_prog} Propagating orbits...')
            batch_data['PVS']   = [prop_func(sat=prop_data.loc[sat], resolution=resolution, duration=duration) for sat in batch_data.index]
            temp_data = sp.extractPropagationData(batch_data, batch_prog=batch_prog)
            temp_data.to_json(f'Temp_Propagation_Data/temp_propagation_data_{batch_no:03}.json')
        
        print('Combining batches...')
        for batch_file in os.listdir('Temp_Propagation_Data'):
            with open(f'Temp_Propagation_Data/{batch_file}') as file:
                batch_data = pd.read_json(file)
            os.remove(f'Temp_Propagation_Data/{batch_file}')
            prop_data.update(batch_data)
        prop_data.to_json('Propagation_Data.json')

        print('Removing temporary files...')
        os.rmdir('Temp_Propagation_Data')
    
    else:
        prop_data['PVS'] = [prop_func(sat=prop_data.loc[sat], resolution=resolution, duration=duration) for sat in prop_data.index]
        sp.extractPropagationData(prop_data).to_json('Propagation_Data.json')


sp = SatellitePropagation()
with open((metadata_file := 'Metadata.json'), 'r') as file:
    metadata = json.load(file)

#runPropagation(sp.propagateTLE, metadata['path']['database'], resolution:=300., duration:=.1)
runPropagation(sp.propagateNumerical, 'Machine_Learning_Testing/Starlink_Data.json', resolution:=300., duration:=.1)

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
    po = PlotOrbit(prop_data_file='Propagation_Data.json')
    po.plotOrbits(metadata_file=metadata_file, limit=500)