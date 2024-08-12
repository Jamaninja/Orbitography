import pandas as pd
import requests
import json
from datetime import datetime, timedelta

from dotenv import load_dotenv
import os
load_dotenv()

directory_path      = "Ephemeris_Database"
ephemeris_data_path = "30_Day_Ephemeris.json"
database_path       = "Database.csv"

def Create_Database():
    if not os.path.exists(f"{directory_path}/{database_path}"):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        if not os.path.exists(f"{directory_path}/{ephemeris_data_path}"):
            download = True
        else:
            mtime = datetime.fromtimestamp(os.path.getmtime(f"{directory_path}/{ephemeris_data_path}"))
            if datetime.now() - mtime > timedelta(days=30): # Redownload epehmeris data if the current data is over 30 days old or # TODO if last update was over 30 days ago
                download = True
            else:
                download = False

        if download:
            uriBase                 = "https://www.space-track.org"
            requestLogin            = "/ajaxauth/login"
            requestCmdAction        = "/basicspacedata/query"
            requestQuery            = "/class/gp/decay_date/null-val/epoch/>now-30/orderby/norad_cat_id/format/json"

            id_st   = os.getenv("id_st")
            pass_st = os.getenv("pass_st")

            with requests.Session() as session:
                login_response = session.post(uriBase + requestLogin, data = {"identity": id_st, "password": pass_st})
                if login_response.status_code != 200:
                    raise Exception(f"{login_response.status_code}\n{login_response.text}")
                
                response = session.get(uriBase + requestCmdAction + requestQuery)
                if response.status_code != 200:
                    raise Exception(f"{response.status_code}\n{response.text}")
                
                with open(f"{directory_path}/{ephemeris_data_path}", "w") as file:
                    file.write(response.text)

        with open(f"{directory_path}/{ephemeris_data_path}", "r") as file:
            data = json.load(file)
            columns = {"CREATION_DATE":     [],
                    "NORAD_CAT_ID":      "",
                    "OBJECT_NAME":       "",
                    "OBJECT_ID":         "",
                    "OBJECT_TYPE":       "",    
                    "RCS_SIZE":          "",
                    "RCS_SIZE_EST":      [], #
                    "MASS_EST":          [], #
                    "COUNTRY_CODE":      "",
                    "LAUNCH_DATE":       "",
                    "SITE":              "",
                    "EPOCH":             [],
                    "MEAN_MOTION":       [],
                    "ECCENTRICITY":      [],
                    "INCLINATION":       [],
                    "RA_OF_ASC_NODE":    [],
                    "ARG_OF_PERICENTER": [],
                    "MEAN_ANOMALY":      [],
                    "REV_AT_EPOCH":      [],
                    "BSTAR":             [],
                    "MEAN_MOTION_DOT":   [],
                    "MEAN_MOTION_DDOT":  [],
                    "SEMIMAJOR_AXIS":    [],
                    "PERIOD":            [],
                    "APOAPSIS":          [],
                    "PERIAPSIS":         [],
                    "DECAY_DATE":        "",
                    "TLE_LINE0":         [],
                    "TLE_LINE1":         [],
                    "TLE_LINE2":         []
            }

            length = len(data)
            df = pd.DataFrame(columns=columns.keys())
            n=0
            constant    = [col for col in columns if type(columns[col]) == str]
            varying     = [col for col in columns if type(columns[col]) == list]
            del varying[1:3]

            for i in data:
                df.loc[n, constant] = [i[x] for x in constant]
                df.loc[n, varying] = [[i[x]] for x in varying]
                df.loc[n, ["RCS_SIZE_EST", "MASS_EST"]] = [[],[]]
                n+=1
                print(f"{n:05d}/{length}")

    df.to_csv(f"{directory_path}/{database_path}")

Create_Database()