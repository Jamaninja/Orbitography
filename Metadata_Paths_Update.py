import json
import os

path = {
    'directory':    'Satellite_Database/',
    'ephemerides':  'Ephemerides.json',
    'database':     'Database.json'
    }
path['ephemerides'] = path['directory'] + path['ephemerides']
path['database']    = path['directory'] + path['database']

if os.path.exists((metadata_file := 'Metadata.json')):
    with open(metadata_file, 'r') as file:
        metadata = json.load(file)
    metadata['path'] = path
    with open(metadata_file, 'w') as file:
        json.dump(metadata, file)
else:
    with open(metadata_file, 'x') as file:
        json.dump({'path': path}, file)