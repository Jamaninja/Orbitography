import os
import pandas as pd
import time
from datetime import datetime
from Orbitography_Functions import DatabaseFunctions

db = DatabaseFunctions()
now_local   = datetime.today()

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

    if os.path.exists(db.path['ephemerides']):
        mtime = datetime.fromtimestamp(os.path.getmtime(db.path['ephemerides']))
        if manual:
            if (mtime_delta := (now_local - mtime).total_seconds()/86400) >= 1:
                mtime_delta = f'{mtime_delta:.1f} days'
            else:
                mtime_delta = f'{mtime_delta*24:.1f} hours'

            print((f'Ephemerides data already exists. Current data is {mtime_delta} old.\n'
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
                    os.rename(db.path['ephemerides'], f'{db.path['ephemerides'].split('.')[0]}_Backup_{mtime.strftime('%Y-%m-%d_%H-%M-%S')}.json')
                case '3':
                    return 3
        
        else: # Automatic update
            os.rename(db.path['ephemerides'], f'{db.path['ephemerides'].split('.')[0]}_Backup_{mtime.strftime('%Y-%m-%d_%H-%M-%S')}.json')

        print('Downloading recent ephemerides data from Space-Track.org.')        
        response = db.downloadEphemerides(date_range=mtime.strftime('%Y-%m-%d%%20%H:%M:%S--now'))
        
    else: # If there is no existing ephemerides data
        print('Downloading 30-day ephemerides data from Space-Track.org.')
        response = db.downloadEphemerides()

    if response == -1:
        print('There have been no ephemerides updates in the requested time interval.')
        return -1
    else:
        return 0

manual = False
backup = True

if updateEphemerides(manual=manual) == -1:
    if manual:
        print('There is no new data to update the database with.\nCancelling update.')
    else:
        # Output log file stating ephemerides update returned blank @ TIME
        pass
else:
    print('Download complete. Updating database.')
    ephem_data = pd.read_json(db.path['ephemerides']).set_index('NORAD_CAT_ID')
    sat_data = pd.read_json(db.path['database'])

    if backup:
        mtime = datetime.fromtimestamp(os.path.getmtime(db.path['database']))
        os.rename(db.path['database'], f'{db.path['database'].split('.')[0]}_Backup_{mtime.strftime('%Y-%m-%d_%H-%M-%S')}.json')

    sat_data.update(ephem_data)
    new_sats = ephem_data[~ephem_data.index.isin(sat_data.index)]
    sat_data = pd.concat([sat_data, new_sats])

    sat_data.to_json(db.path['database'])
    sat_data.to_csv(f'{db.path['database'].split('.')[0]}.csv')
    print('Database updated.')

    output_log = ('Update Log.'                            # TODO: Finish writing output log
                 f'Date: {now_local.strftime('%Y-%m-%d')}'
                 f'Time: {now_local.strftime('%H:%M:%S UTC')}'
                 f'{len(ephem_data)} objects added.')

input('Press enter to exit.')