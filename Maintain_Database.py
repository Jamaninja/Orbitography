import pandas as pd
import requests
import json
from datetime import datetime, timedelta, UTC
import time
import sys

from dotenv import load_dotenv
import os
load_dotenv()

columns = {'OBJECT_NAME':       '', # Not all object information is initially known, so despite being "constant", these values may still be updated over time 
           'OBJECT_ID':         '', # COSPAR ID
           'OBJECT_TYPE':       '', # PAYLOAD, ROCKET BODY, DEBRIS, UNKNOWN, or OTHER
           'RCS_SIZE':          '',
           'RCS_SIZE_EST':      {}, # RCS size and satellite mass can both be predicted using machine learning. These columns are preemptively added for later use
           'MASS_EST':          {}, # They will store an array of estimated values, as well as the statistical mean (assuming all estimated values are normally distributed) 
           'COUNTRY_CODE':      '',
           'LAUNCH_DATE':       '',
           'SITE':              '',
           'EPOCH':             [], # Orbital history data is stored, not overwriten, to enable more accurate ML estimates for RCS size and satellite mass
           'MEAN_MOTION':       [], # History will be deleted after set number of updates, to prevent database bloat
           'ECCENTRICITY':      [],
           'INCLINATION':       [],
           'RA_OF_ASC_NODE':    [],
           'ARG_OF_PERICENTER': [],
           'MEAN_ANOMALY':      [],
           'REV_AT_EPOCH':      [],
           'BSTAR':             [],
           'MEAN_MOTION_DOT':   [],
           'MEAN_MOTION_DDOT':  [],
           'SEMIMAJOR_AXIS':    [],
           'PERIOD':            [],
           'APOAPSIS':          [],
           'PERIAPSIS':         [],
           'DECAY_DATE':        '',
           'TLE_LINE0':         '', # TLE_LINE0
           'TLE_LINE1':         [],
           'TLE_LINE2':         [],
           'CREATION_DATE':     []
           }
constant    = [col for col in columns if type(columns[col]) == str]     # constant and varying are perhaps not the most accurate names, but they suffice
varying     = [col for col in columns if type(columns[col]) == list]

def spaceTrackRequest(requestQuery):
    '''
    Sends a query to Space-Track.org and returns the response

    Args:
        requestQuery: string
            The desired Space-Track API query
    
    Returns:
        response.text: list[dict]
            A list of for all ephemerides for objects in orbit that have been updated in the requested time period, loaded from the json object received from Space-Track
    '''

    uriBase                 = 'https://www.space-track.org'
    requestLogin            = '/ajaxauth/login'
    requestCmdAction        = '/basicspacedata/query'

    id_st   = os.getenv('id_st')
    pass_st = os.getenv('pass_st')

    with requests.Session() as session:
        if (login_response := session.post(uriBase + requestLogin, data = {'identity': id_st, 'password': pass_st})).status_code != 200:
            raise Exception(f'{login_response.status_code}\n{login_response.text}')
        
        if (response := session.get(uriBase + requestCmdAction + requestQuery)).status_code != 200:
            raise Exception(f'{response.status_code}\n{response.text}')

    return json.loads(response.text)
                   
def checkDatabase(ephemerides_response):
    '''
    Checks if any ephemerides data is present. If not, it downloads a 30 day ephemerides dataset. Else, gives the user the option of overwriting or backing up the current dataset,
    before downloading all ephemerides updates since the last update.

    Returns:
        -1:
            Database does not need updating, signals to terminate database update
        0:
            User wishes the update the database, signals to update the database from ephemerides data
        2:
            User wishes to create a backup of the database, signals to backup current database before updating
    '''

    if os.path.exists(path['database']): # Checks if the database already exists, then asks the user how they would like to proceed if it does
        ctime = datetime.fromtimestamp(os.path.getctime(path['database']))
        if (ctime_delta := (now - ctime).total_seconds()/86400) >= 1:
            ctime_delta = f'{ctime_delta:.1f} days'
        else:
            ctime_delta = f'{ctime_delta*24:.1f} hours'

        print((f'{path['database']} already exists. The database was last updated {ctime_delta} ago.\n'
                'Would you like to:'))
        while (database_case := input((
             '[1] Update the database\n'
             ' 2  Create a backup of the database and create a new one\n'
             ' 3  Do not update the database\n'
             ' >  ')).strip() or '1') not in ['1', '2', '3']:
            
            time.sleep(0.5)
            print('Please eneter an option 1 - 3:')
            
        match database_case:
            case '2':
                return 2
            case '3':
                time.sleep(0.5)
                print('Cancelling process.')
                return -1

    else:
        createDatabase(ephemerides_response)
        return -1
    
    return 0

def createDatabase(ephemerides_response):
    '''
    Creates an a database of orbital objects, from new or pre-existing ephemerides data from Space-Track.org
    Updates the Metadata.json file with the file and directory paths, creating the file if required
    '''

    if not os.path.exists(path['directory']):
        os.makedirs(path['directory'])
    
    match ephemerides_response:
        case 0:
            print('Downloading 30-day ephemerides data from Space-Track.org.')
            downloadEphemerides()
        case -1:
            Exception('There is no ephemerides data to create a database from.')
            return -1
        
    ephem_data  = pd.read_json(path['ephemerides']).set_index('NORAD_CAT_ID')
    sat_data    = pd.DataFrame(index = ephem_data.index, columns=columns.keys())
    sat_data.update(ephem_data)
    sat_data.update(sat_data[varying].map(lambda x: [x]))
    sat_data.update(sat_data[['RCS_SIZE_EST', 'MASS_EST']].map(lambda x: [{'size': 0, 'size_est': []}]))

    sat_data.to_json(path['database'])
    sat_data.to_csv(f'{path['database'].split('.')[0]}.csv')

    updateMetadata()

    return 0

def downloadEphemerides(**kwargs):
    '''
    Downloads the latest ephemerides within the requested time range

    Args:
        **kwargs
            date_range: string
                The requested updated period, formatted to Space-Track's API. Defaults to the previous 30 days
    
    Returns:
        -1:
            Indicates that there is no new ephemerides data, signals to terminate database update
        response: list[dict]
            A list of for all ephemerides for objects in orbit that have been updated in the requested time period
    '''

    date_range = kwargs.get('date_range') or '>now-30'
    if (response := spaceTrackRequest(f'/class/gp/decay_date/null-val/epoch/{date_range}/orderby/norad_cat_id/format/json')) == []:
        print('There have been no changes since the previous update.')
        return -1
    else:
        with open(path['ephemerides'], 'w') as file:
            json.dump(response, file)
        return response
    
def updateEphemerides():
    '''
    Checks if any ephemerides data is present. If not, it downloads a 30 day ephemerides dataset. Else, gives the user the option of overwriting or backing up the current dataset,
    before downloading all ephemerides updates since the last update.

    Returns:
        -1:
            Indicates that there is no new ephemerides data, signals to terminate database update
        0:
            Update the database with new ephemerides data, signals to download ephemerides data dating back to last update
        3:
            Update the database with existing ephemerides data, signals to not download new ephemerides data
    '''

    if os.path.exists(path['ephemerides']):
        ctime = datetime.fromtimestamp(os.path.getctime(path['ephemerides']))
        if (ctime_delta := (now - ctime).total_seconds()/86400) >= 1:
            ctime_delta = f'{ctime_delta:.1f} days'
        else:
            ctime_delta = f'{ctime_delta*24:.1f} hours'
        
        print((f'Ephemerides data already exists. Current data is {ctime_delta} old.\n'
                'Would you like to:'))
        while (epheremides_case := input((
             '[1] Overwrite the ephemerides data with updated data\n'
            f' 2  Backup the current ephemerides data before updating\n'
             ' 3  Proceed with the current ephemerides data\n'
             ' >  ')).strip() or '1') not in ['1', '2', '3']:
            
            time.sleep(0.5)
            print('Please eneter an option 1 - 3:')
        
        match epheremides_case:
            case '2':
                os.rename(path['ephemerides'], f'{path['ephemerides'].split('.')[0]}_Backup_{now.strftime('%Y-%m-%d_%H-%M-%S')}.json')
            case '3':
                return 3 

        response = downloadEphemerides(date_range=ctime.strftime('%Y-%m-%d%%20%H:%M:%S--now'))
    
    else:
        response = downloadEphemerides()
    
    if response == -1:
        return -1
    else:
        return 0
    
def updateDatabase():
    match (ephemerides_response := updateEphemerides()):
        case -1:
            print('There is no new data to update the database with.')
            return -1
        case _:
            ephem_data = pd.read_json(path['ephemerides'])

    match checkDatabase(ephemerides_response):
        case -1:
            return -1
        case 0:
            sat_data = pd.read_json(path['database'])
        case 2:
            sat_data = pd.read_json(path['database'])
            os.rename(path['database'], f'{path['database'].split('.')[0]}_Backup_{now.strftime('%Y-%m-%d_%H-%M-%S')}.json')

    length      = len(ephem_data)
    time_start  = datetime.today()
    
    update = ephem_data.loc[:, varying][ephem_data.loc[:, 'NORAD_CAT_ID'].isin(sat_data.loc[:, 'NORAD_CAT_ID'])].set_index('NORAD_CAT_ID')
    sat_data.loc[:, varying] = sat_data.loc[:, varying].apply(lambda xs: list(map(lambda x: xs.insert(0, x), xs)))
    
    add = ephem_data[~ephem_data.loc[:, 'NORAD_CAT_ID'].isin(update.loc[:, 'NORAD_CAT_ID'])]

        
'''
    for index, ephemeris in enumerate(ephemerides):
        values = [value for key, value in ephemeris.items() if key in varying]
        for n, i in enumerate(sat_data.loc[index, varying]):
            i.insert(0, values[n])
            if len(i) > database_limit:
                del i[database_limit]

        if not index % 50:
            print(f'Creating database:  {index/length:06.2%}    Elapsed time: {str(datetime.today() - time_start)[2:11]}')

    print(f'Creating database: 100.00%    Elapsed time: {str(datetime.today() - time_start)[2:11]}')
    print('Saving database...')'''
    #sat_data.to_json(path['database']['path'])
    #sat_data.to_csv(f'{path['database']['path'].split('.')[0]}.csv')

def updateMetadata():
    '''
    Updates the Metadata.json file with the file and directory paths, creating the file if required
    '''

    if os.path.exists((metadata_file := 'Metadata.json')):
        with open(metadata_file, 'r') as file:
            metadata = json.load(file)
        metadata['path'] = path
        with open(metadata_file, 'w') as file:
            json.dump(metadata, file)
    else:
        with open(metadata_file, 'x') as file:
            json.dump({'path': path}, file)

path = {
    'directory':    'Satellite_Database/',
    'ephemerides':  'Ephemerides.json',
    'database':     'Database.json'
    }
path['ephemerides'] = path['directory'] + path['ephemerides']
path['database']    = path['directory'] + path['database']

database_limit = 10
now = datetime.today()
#createDatabase(0)
#updateDatabase()