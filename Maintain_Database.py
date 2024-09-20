import pandas as pd
import requests
import json
from datetime import datetime, timedelta, UTC
import time
import sys

from dotenv import load_dotenv
import os
load_dotenv()

columns = {'CREATION_DATE':     [],
            'NORAD_CAT_ID':      '',
            'OBJECT_NAME':       '',
            'OBJECT_ID':         '', # COSPAR ID
            'OBJECT_TYPE':       '', # PAYLOAD, ROCKET BODY, DEBRIS, UNKNOWN, or OTHER
            'RCS_SIZE':          '',
            'RCS_SIZE_EST':      {},
            'MASS_EST':          {},
            'COUNTRY_CODE':      '',
            'LAUNCH_DATE':       '',
            'SITE':              '',
            'EPOCH':             [],
            'MEAN_MOTION':       [],
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
            'TLE_LINE0':         '',
            'TLE_LINE1':         [],
            'TLE_LINE2':         []
            }
constant    = [col for col in columns if type(columns[col]) == str]
varying     = [col for col in columns if type(columns[col]) == list]

def spaceTrackRequest(requestQuery):
    uriBase                 = 'https://www.space-track.org'
    requestLogin            = '/ajaxauth/login'
    requestCmdAction        = '/basicspacedata/query'

    id_st   = os.getenv('id_st')
    pass_st = os.getenv('pass_st')

    with requests.Session() as session:
        login_response = session.post(uriBase + requestLogin, data = {'identity': id_st, 'password': pass_st})
        if login_response.status_code != 200:
            raise Exception(f'{login_response.status_code}\n{login_response.text}')
                
        response = session.get(uriBase + requestCmdAction + requestQuery)
    if response.status_code != 200:
        raise Exception(f'{response.status_code}\n{response.text}')

    return response.text

def updateEphemerides():
    if os.path.exists(path['ephemerides']['path']):
        mtime = datetime.fromtimestamp(os.path.getmtime(path['ephemerides']['path']))
        mtime_delta = f'{(now - mtime).total_seconds()/86400:.1f}'
        
        epheremis_case = input((
            f'Ephemerides data already exists. Current data is {mtime_delta} days old.\n'
             'Would you like to:\n'
             '[1] Overwrite the ephemerides data\n'
            f' 2  Create a backup of the current ephemerides data and download a new dataset from the prior {mtime_delta} days\n'
             ' 3  Proceed with the current ephemerides data\n'
             ' >  ')).strip() or '1'
        
        while True:
            match epheremis_case:
                case '1':
                    break

                case '2':
                    os.rename(path['ephemerides']['path'], f'{path['directory']}{path['ephemerides']['file']}_Backup_{now.strftime('%Y-%m-%d_%H-%M-%S')}{path['ephemerides']['extn']}')
                    break
                
                case '3':
                    return 0

                case _:
                    epheremis_case = input((
                         'Please enter an option 1 - 3\n'
                         '[1] Overwrite the ephemerides data\n'
                        f' 2  Create a backup of the current ephemerides data and download a new dataset from the prior {mtime_delta} days\n'
                         ' 3  Proceed with the current ephemerides data\n'
                         ' >  ')).strip() or '1'
                    
        date_range = mtime.strftime('%Y-%m-%d%%20%H:%M:%S--now')
        if (response := spaceTrackRequest(f'/class/gp/decay_date/null-val/epoch/{date_range}/orderby/norad_cat_id/format/json')) == '[]':
            print('There have been no changes since the previous update.')
            return -1
        else:
            with open(path['ephemerides']['path'], 'w') as file:
                file.write(response)
            return 0
    
    else:
        with open(path['ephemerides']['path'], 'w') as file:
            file.write(spaceTrackRequest('/class/gp/decay_date/null-val/epoch/>now-30/orderby/norad_cat_id/format/json'))
        return 0
                    
def checkDatabase():
    if os.path.exists(path['database']['path']): # Checks if the database already exists, then asks the user how they would like to proceed if it does
        mtime = datetime.fromtimestamp(os.path.getmtime(path['database']['path']))
        mtime_delta = f'{(now - mtime).total_seconds()/86400:.1f}'

        database_case = input((
            f'{path['database']} already exists. The database was last updated {mtime_delta} days ago.\n'
             'Would you like to:\n'
             '[1] Update the database\n'
             ' 2  Create a backup of the database and create a new one\n'
             ' 3  Do not update the database\n'
             ' >  ')).strip() or '1'

        while True:
            match database_case:
                case '1':
                    return 0
                
                case '2':
                    os.rename(path['database']['path'], f'{path['directory']}{path['database']['file']}_Backup_{now.strftime('%Y-%m-%d_%H-%M-%S')}{path['database']['extn']}')
                    return 0
                
                case '3':
                    return -1

                case _:
                    database_case = input((
                        'Please enter an option 1 - 3\n'
                        '[1] Update the database\n'
                        ' 2  Create a backup of the database and create a new one\n'
                        ' 3  Do not update the database\n'
                        ' >  ')).strip() or '1'

    else:
        createDatabase()
        return 0

def createDatabase():
    if not os.path.exists(path['directory']):
        os.makedirs(path['directory'])
    
    with open(path['ephemerides']['path'], 'w') as file:
        file.write((response:= spaceTrackRequest('/class/gp/decay_date/null-val/epoch/>now-30/orderby/norad_cat_id/format/json')))
    ephemerides = json.loads(response)

    sat_data    = pd.DataFrame(columns=columns.keys())
    n, length   = 0, len(ephemerides)
    time_start  = datetime.today()

    for ephemeris in ephemerides:
        sat_data.loc[n, constant] = [ephemeris[x] for x in constant]
        sat_data.loc[n, varying] = [[ephemeris[x]] for x in varying]
        sat_data.loc[n, ['RCS_SIZE_EST', 'MASS_EST']] = [{'size': 0, 'size_est': []},{'mass': 0, 'mass_est': []}]
        n += 1
        if not n % 50:
            print(f'Creating database:  {n/length:06.2%}    Elapsed time: {str(datetime.today() - time_start)[2:11]}')

    print(f'Creating database: 100.00%    Elapsed time: {str(datetime.today() - time_start)[2:11]}')
    print('Saving database...')
    sat_data.to_json(path['database']['path'])
    sat_data.to_csv(f'{path['database']['path'].split('.')[0]}.csv')

def updateDatabase():
    '''if updateEphemerides() == -1:
        return -1
    if checkDatabase() == -1:
        return -1
    '''
    sat_data = pd.read_json(path['database']['path'])
    with open(path['ephemerides']['path'], 'r') as file:
        ephemerides = json.load(file)

    for index, ephemeris in enumerate(ephemerides):
        values = [value for key, value in ephemeris.items() if key in varying]
        for n, i in enumerate(sat_data.loc[index, varying]):
            i.insert(0, values[n])


path = {
    'directory':    'Satellite_Database/',
    'ephemerides': {'file': 'Ephemerides',
                    'extn': '.json'},
    'database':    {'file': 'Database',
                    'extn': '.json'}
    }
path['ephemerides']['path'] = f'{path['directory']}{path['ephemerides']['file']}{path['ephemerides']['extn']}'
path['database']['path']    = f'{path['directory']}{path['database']['file']}{path['database']['extn']}'

database_limit = 10
now = datetime.today()
createDatabase()
#updateDatabase()