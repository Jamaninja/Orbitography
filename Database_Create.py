import os
import pandas as pd
from Orbitography_Functions import DatabaseFunctions

'''
Creates an a database of orbital objects, from new or pre-existing ephemerides data from Space-Track.org
'''

columns = ['OBJECT_NAME',
           'OBJECT_ID',     # COSPAR ID
           'OBJECT_TYPE',   # PAYLOAD, ROCKET BODY, DEBRIS, UNKNOWN, or OTHER
           'RCS_SIZE',
           'RCS_SIZE_EST',  # RCS size and satellite mass can both be predicted using machine learning. These columns are preemptively added for later use
           'MASS_EST',      # They will store an array of estimated values, as well as the statistical mean (assuming all estimated values are normally distributed)
           'COUNTRY_CODE',
           'LAUNCH_DATE',
           'SITE',
           'EPOCH',
           'MEAN_MOTION',
           'ECCENTRICITY',
           'INCLINATION',
           'RA_OF_ASC_NODE',
           'ARG_OF_PERICENTER',
           'MEAN_ANOMALY',
           'REV_AT_EPOCH',
           'BSTAR',
           'MEAN_MOTION_DOT',
           'MEAN_MOTION_DDOT', 
           'SEMIMAJOR_AXIS',
           'PERIOD',
           'APOAPSIS',
           'PERIAPSIS',
           'DECAY_DATE',
           'TLE_LINE0',
           'TLE_LINE1',
           'TLE_LINE2',
           'CREATION_DATE']

db = DatabaseFunctions()

if not os.path.exists(db.path['directory']):
    os.makedirs(db.path['directory'])

print('Downloading 30-day ephemerides data from Space-Track.org.')
if db.downloadEphemerides() == -1:
    Exception('There is no ephemerides data to create a database from.')
    
else:
    print('Download complete. Creating database.')
    ephem_data  = pd.read_json(db.path['ephemerides']).set_index('NORAD_CAT_ID')
    sat_data    = pd.DataFrame(index = ephem_data.index, columns=columns)
    sat_data.update(ephem_data)
    #sat_data.update(sat_data[['RCS_SIZE_EST', 'MASS_EST']].map(lambda x: [{'size': 0, 'size_est': []}]))

    sat_data.to_json(db.path['database'])
    sat_data.to_csv(f'{db.path['database'].split('.')[0]}.csv')
    print('Database created.')

input('Press enter to exit.')