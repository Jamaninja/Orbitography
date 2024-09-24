import pandas as pd
import requests
import json
from datetime import datetime, timedelta, UTC
import time
import sys

from dotenv import load_dotenv
import os
load_dotenv()

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
                   
def createDatabase():
    '''
    Creates an a database of orbital objects, from new or pre-existing ephemerides data from Space-Track.org
    Updates the Metadata.json file with the file and directory paths, creating the file if required
    '''

    if not os.path.exists(path['directory']):
        os.makedirs(path['directory'])
    
    print('Downloading 30-day ephemerides data from Space-Track.org.')
    if downloadEphemerides() == -1:
        Exception('There is no ephemerides data to create a database from.')
        return -1
        
    ephem_data  = pd.read_json(path['ephemerides']).set_index('NORAD_CAT_ID')
    sat_data    = pd.DataFrame(index = ephem_data.index, columns=columns)
    sat_data.update(ephem_data)
    sat_data.update(sat_data[['RCS_SIZE_EST', 'MASS_EST']].map(lambda x: [{'size': 0, 'size_est': []}]))

    sat_data.to_json(path['database'])
    sat_data.to_csv(f'{path['database'].split('.')[0]}.csv')

    updateMetadata()

    return 0

def downloadEphemerides(date_range='>now-30'):
    '''
    Downloads the latest ephemerides within the requested time range

    Args:
        date_range: string
            The requested updated period, formatted to Space-Track's API. Defaults to the previous 30 days
    
    Returns:
        -1:
            Indicates that there is no new ephemerides data, signals to terminate database update
        response: list[dict]
            A list of for all ephemerides for objects in orbit that have been updated in the requested time period
    '''

    if (response := spaceTrackRequest(f'/class/gp/decay_date/null-val/epoch/{date_range}/orderby/norad_cat_id/format/json')) == []:
        return -1
    else:
        with open(path['ephemerides'], 'w') as file:
            json.dump(response, file)
        return response

def updateEphemerides(manual=False):
    '''
    Checks if any ephemerides data is present. If not, it downloads a 30 day ephemerides dataset. Else, gives the user the option of overwriting or backing up the current dataset,
    before downloading all ephemerides updates since the last update.

    Returns:
        0:
            Update the database with new ephemerides data, signals to download ephemerides data dating back to last update
        3:
            Update the database with existing ephemerides data, signals to not download new ephemerides data
        -1:
            Indicates that there is no new ephemerides data, signals to terminate database update
    '''
    if manual:
        if os.path.exists(path['ephemerides']):
            ctime = datetime.fromtimestamp(os.path.getctime(path['ephemerides']))
            if (ctime_delta := (now - ctime).total_seconds()/86400) >= 1:
                ctime_delta = f'{ctime_delta:.1f} days'
            else:
                ctime_delta = f'{ctime_delta*24:.1f} hours'
            
            print((f'Ephemerides data already exists. Current data is {ctime_delta} old.\n'
                    'Would you like to:'))
            while (epheremides_case := input((
                '[1] Backup the current ephemerides data before updating\n'
               f' 2  Overwrite the ephemerides data with updated data\n'
                ' 3  Proceed with the current ephemerides data\n'
                ' >  ')).strip() or '1') not in ['1', '2', '3']:
                
                time.sleep(0.5)
                print('Please eneter an option 1 - 3:')
            
            match epheremides_case:
                case '1':
                    os.rename(path['ephemerides'], f'{path['ephemerides'].split('.')[0]}_Backup_{now.strftime('%Y-%m-%d_%H-%M-%S')}.json')
                case '3':
                    return 3 

            response = downloadEphemerides(date_range=ctime.strftime('%Y-%m-%d%%20%H:%M:%S--now'))
        
        else:
            response = downloadEphemerides()
        
        if response == -1:
            print('There have been no ephemerides updates in the requested time interval.')
            return -1
        else:
            return 0
    
    else:
        if os.path.exists(path['ephemerides']):
            ctime = datetime.fromtimestamp(os.path.getctime(path['ephemerides']))
            os.rename(path['ephemerides'], f'{path['ephemerides'].split('.')[0]}_Backup_{now.strftime('%Y-%m-%d_%H-%M-%S')}.json')
            response = downloadEphemerides(date_range=ctime.strftime('%Y-%m-%d%%20%H:%M:%S--now'))
        else:
            response = downloadEphemerides()
        
        if response == -1:
            return -1
        else:
            return 0

def updateDatabase(manual=False, backup=True):
    if updateEphemerides(manual=manual) == -1:
        if manual:
            print('There is no new data to update the database with.\nCancelling update.')
        else:
            # Output log file stating ephemerides update returned blank @ TIME
            pass
        return -1
    
    ephem_data = pd.read_json(path['ephemerides']).set_index('NORAD_CAT_ID')
    sat_data = pd.read_json(path['database'])

    if backup:
        os.rename(path['database'], f'{path['database'].split('.')[0]}_Backup_{now.strftime('%Y-%m-%d_%H-%M-%S')}.json')

    sat_data.update(ephem_data)
    new_sats = ephem_data[~ephem_data.index.isin(sat_data.index)]
    sat_data = pd.concat([sat_data, new_sats])

    sat_data.to_json(path['database']['path'])
    sat_data.to_csv(f'{path['database']['path'].split('.')[0]}.csv')

    output_log = ( 'Update Log.'                            # TODO: Finish writing output log
                  f'Date: {now.strftime('%Y-%m-%d')}'
                  f'Time: {now.strftime('%H:%M:%S')}'
                  f'{len(ephem_data)} objects added.')

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
#createDatabase()
updateDatabase()