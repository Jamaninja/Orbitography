import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import sys

from dotenv import load_dotenv
import os
load_dotenv()

def spaceTrackRequest(requestQuery):
    uriBase                 = "https://www.space-track.org"
    requestLogin            = "/ajaxauth/login"
    requestCmdAction        = "/basicspacedata/query"

    id_st   = os.getenv("id_st")
    pass_st = os.getenv("pass_st")

    with requests.Session() as session:
        login_response = session.post(uriBase + requestLogin, data = {"identity": id_st, "password": pass_st})
        if login_response.status_code != 200:
            raise Exception(f"{login_response.status_code}\n{login_response.text}")
                
        response = session.get(uriBase + requestCmdAction + requestQuery)
    if response.status_code != 200:
        raise Exception(f"{response.status_code}\n{response.text}")

    return response.text

def checkEphemerisPath(path):
    if os.path.exists(f"{path["directory"]}/{path["ephemeris"]}"):
        mtime_pass = datetime.now() - datetime.fromtimestamp(os.path.getmtime(f"{path["directory"]}/{path["ephemeris"]}"))
        new_ephemeris_path = f"{path["ephemeris"].split(".")[0]}_2.{path["ephemeris"].split(".")[1]}" # TODO: Check if a database with the new name exists, and iterate respectively
        
        epheremis_exist_case = input((
            f"Ephemeris data already exists. Current data is {(mtime_pass.total_seconds())/86400:.1f} days old.\n"
             "Would you like to:\n"
            f" 1  Keep the current ephemeris data, but download a new dataset \"{new_ephemeris_path}\" from the past 30 days to create the database from\n"
             "[2] Delete the current data and download a new dataset\n"
             " 3  Keep the current data and do not download a new dataset\n"
             " >  ")).strip() or "2"
        
        while True:
            match epheremis_exist_case:
                case "1":
                    print("Downloading 30 day ephemeris data from SpaceTrack.")
                    return new_ephemeris_path
                    
                case "2":
                    os.remove(f"{path["directory"]}/{path["ephemeris"]}")
                    print("Downloading 30 day ephemeris data from SpaceTrack.")
                    return
                
                case "3":
                    return

                case _:
                    epheremis_exist_case = input((
                         "Please enter a number 1 - 3\n"
                        f" 1  Keep the current ephemeris data, but download a new dataset \"{new_ephemeris_path}\" from the past 30 days to create the database from\n"
                         "[2] Delete the current data and download a new dataset\n"
                         " 3  Keep the current data and do not download a new dataset\n"
                         " >  ")).strip() or "2"

def checkDatabasePath(path):
    if os.path.exists(f"{path["directory"]}/{path["database"]}"): # Checks if the database already exists, then asks the user how they would like to proceed if it does
        mtime_pass = datetime.now() - datetime.fromtimestamp(os.path.getmtime(f"{path["directory"]}/{path["database"]}"))
        new_database_path = f"{path["database"].split(".")[0]}_2.{path["database"].split(".")[1]}" # TODO: Check if a database with the new name exists, and iterate respectively

        database_exist_case = input((
            f"{path["database"]} already exists. The database was last updated {(mtime_pass.total_seconds())/86400:.1f} days ago.\n"
             "Would you like to:\n"
            f"[1] Create a new database \"{new_database_path}\"\n"
             " 2  Delete the current database and create a new one\n"
             " 3  Cancel creating a database\n"
             " >  ")).strip() or "1"

        while True:
            match database_exist_case:
                case "1":
                    print("Creating database")
                    return new_database_path
                
                case "2":
                    os.remove(f"{path["directory"]}/{path["database"]}")
                    print("Creating database")
                    return
                
                case "3":
                    print("Terminating database update")
                    sys.exit(0)

                case _:
                    database_exist_case = input((
                        f"Please enter a number 1 - 3\n"
                        f"[1] Create a new database \"{new_database_path}\"\n"
                        " 2  Overwrite the current database\n"
                        " 3  Terminate the database update process\n"
                        " >  ")).strip() or "1"

def downloadEphemeris(path):
    '''uriBase                 = "https://www.space-track.org"
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
                
        with open(f"{path["directory"]}/{path["ephemeris"]}", "w") as file:
            file.write(response.text)'''
    
    with open(f"{path["directory"]}/{path["ephemeris"]}", "w") as file:
        file.write(spaceTrackRequest("/class/gp/decay_date/null-val/epoch/>now-30/orderby/norad_cat_id/format/json"))

def createDatabase(path):
    if not os.path.exists(path["directory"]):
        os.makedirs(path["directory"])
    if (new_ephemeris_path := checkEphemerisPath(path)):
        path["ephemeris"] = new_ephemeris_path
    if (new_database_path := checkDatabasePath(path)):
        path["database"] = new_database_path
        
    downloadEphemeris(path)

    with open(f"{path["directory"]}/{path["ephemeris"]}", "r") as file:
        data = json.load(file)

    columns = {"CREATION_DATE":     [],
               "NORAD_CAT_ID":      "", # COSPAR (INTLDES) ID also required, need to find how to get data from Space-Track
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
    constant    = [col for col in columns if type(columns[col]) == str]
    varying     = [col for col in columns if type(columns[col]) == list]
    del varying[1:3]
    n = 0

    for i in data:
        df.loc[n, constant] = [i[x] for x in constant]
        df.loc[n, varying] = [[i[x]] for x in varying]
        df.loc[n, ["RCS_SIZE_EST", "MASS_EST"]] = [[],[]]
        n += 1
        if not n % 50:
            print(f"Creating database:  {n/length::06.2%}%")

    print("Creating database: 100.00%")
    print("Saving database.")
    df.to_pickle(f"{path["directory"]}/{path["database"]}")
    df.to_csv(f"{path["directory"]}/{path["database"][:-3]}csv")
    return path

def updateDatabase(path):
    mtime = datetime.fromtimestamp(os.path.getmtime(f"{path["directory"]}/30_Day_Ephemeris.json"))
    with open(f"{path["directory"]}/{"update_test.json"}", "w") as file:
        file.write(spaceTrackRequest(f"/class/gp/decay_date/null-val/epoch/>{mtime.strftime("%Y-%m-%d")}%20{mtime.strftime("%H:%M:%S")}/orderby/norad_cat_id/format/json"))

path = {
    "directory":    "Ephemeris_Database",
    "ephemeris":    "Initial_Ephemeris.json",
    "database":     "Database.pkl"
    }

createDatabase(path)