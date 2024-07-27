import pandas as pd
from spacetrack import SpaceTrackClient
from dotenv import load_dotenv
import os
load_dotenv()

from Orbitography_Functions import now

sat_data = {
    'ENVISAT': {
        'norad_id': 27386,  # For Space-Track TLE queries
        'cospar_id': '0200901',  # For laser ranging data queries
        'sic_id': '6179',  # For writing in CPF files
        'mass': 8000.0, # kg; TODO: compute proper value
        'cross_section': 100.0, # m2; TODO: compute proper value
        'cd': 2.0, # TODO: compute proper value
        'cr': 1.0  # TODO: compute proper value
    },
    'LAGEOS2': {
        'norad_id': 22195,
        'cospar_id': '9207002',
        'sic_id': '5986',
        'mass': 405.0, # kg
        'cross_section': 0.2827, # m2
        'cd': 2.0, # TODO: compute proper value
        'cr': 1.0  # TODO: compute proper value
    },
    'TECHNOSAT': {
        'norad_id': 42829,
        'cospar_id': '1704205',
        'sic_id': '6203',
        'mass': 20.0, # kg
        'cross_section': 0.10, # m2,
        'cd': 2.0, # TODO: compute proper value
        'cr': 1.0  # TODO: compute proper value
    },
    'SNET1': {
        'norad_id': 43189,
        'cospar_id': '1801410',
        'sic_id': '6204',
        'mass': 8.0, # kg
        'cross_section': 0.07, # m2
        'cd': 2.0, # TODO: compute proper value
        'cr': 1.0  # TODO: compute proper value
    },
    'ISS (ZARYA)': {
        'norad_id': 25544,
        'cospar_id': '1998-067A',
        'sic_id': '0',
        'mass': 450_000.0, # kg
        'cross_section': 2_481.0, # m2
        'cd': 2.0, # TODO: compute proper value
        'cr': 1.0  # TODO: compute proper value
    }
}

id_st = os.getenv('id_st')
pass_st = os.getenv('pass_st')
st = SpaceTrackClient(identity=id_st, password=pass_st)

for sat in sat_data:
    tle = st.tle(norad_cat_id=sat_data[sat]['norad_id'], epoch='<{}'.format(now), orderby='epoch desc', limit=1, format='tle').split('\n')
    sat_data[sat]['TLE'] = (tle[0],tle[1])

sat_dataframe = pd.DataFrame(sat_data)
sat_dataframe.to_pickle('Satellite_Data.pkl')