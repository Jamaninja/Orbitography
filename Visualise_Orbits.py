import orekit
vm = orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir, absolutedate_to_datetime, datetime_to_absolutedate
setup_orekit_curdir()

from org.orekit.propagation.analytical.tle import TLE, TLEPropagator # type: ignore
from Orbitography_Functions import PlotFunctions, SatelliteFunctions

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from moviepy.editor import VideoClip
import io 
from PIL import Image
import ffmpeg
from subprocess import Popen, PIPE

sf = SatelliteFunctions()

with open("tle_short.txt") as file:
    tle = file.readlines()

tle1, tle2 = tle[::2], tle[1::2]
tle_info, pos_info = {}, {}

for i,_ in enumerate(tle1):
    tle = TLE(tle1[i],tle2[i])
    if (sat := str(tle).split("\n")[2].split(" ")[1]) not in tle_info:
        tle_info[sat] = pd.DataFrame(columns=["tle", "epoch", "start", "end", "times"])
        pos_info[sat] = pd.DataFrame(columns=["time", "lat", "lon"])
        n = i

    tle_info[sat].loc[i-n, "tle"]   = tle
    tle_info[sat].loc[i-n, "epoch"] = absolutedate_to_datetime(tle.getDate())
    tle_info[sat].loc[i-n, "times"] = []

sat_epoch = lambda ind: tle_info["40697"].loc[ind, "epoch"] if ind >=0 else tle_info["40697"].loc[tle_info["40697"].index[ind], "epoch"]

res         = 1 # minutes per frame
time_start  = sat_epoch(0)
time_end    = sat_epoch(-1)
time_start  = datetime(time_start.year, time_start.month, time_start.day, 0, 0, 0, 0) 
time_end    = datetime(time_end.year, time_end.month, time_end.day, 0, 0, 0, 0) + timedelta(days=1)
time_array  = [time_start + timedelta(minutes=m) for m in range(0, int((time_end-time_start).total_seconds()/60 + 1), res)]

pvs = []
pos_info["40697"].loc[:, "time"] = time_array 

for i in tle_info["40697"].index:
    if not i:
        tle_info["40697"].loc[0, "start"] = time_start
        tle_info["40697"].loc[0, "end"]   = sat_epoch(0) + (sat_epoch(1) - sat_epoch(0)) / 2
        
    elif i == tle_info["40697"].index[-1]:
        tle_info["40697"].loc[i, "start"] = sat_epoch(-2) + (sat_epoch(-1) - sat_epoch(-2)) / 2
        tle_info["40697"].loc[i, "end"]   = time_end
        
    else:
        tle_info["40697"].loc[i, "start"] = sat_epoch(i-1) + (sat_epoch(i) - sat_epoch(i-1)) / 2
        tle_info["40697"].loc[i, "end"]   = sat_epoch(i) + (sat_epoch(i+1) - sat_epoch(i)) / 2

    for t in time_array:
        if tle_info["40697"].loc[i, "start"] <= t < tle_info["40697"].loc[i, "end"]:
            tle_info["40697"].loc[i, "times"].append(t)

        elif t > tle_info["40697"].loc[i, "end"]:
            break

        else:
            tle_info["40697"].loc[i, "times"].append(t) # Added seperately to prevent a 1 in 3.6 billion odds crash
    
    time_array = [x for x in time_array if x not in tle_info["40697"].loc[i, "times"]]

    propagator = TLEPropagator.selectExtrapolator(tle_info["40697"].loc[i, "tle"])
    for i in [propagator.propagate(datetime_to_absolutedate(t)).getPVCoordinates() for t in tle_info["40697"].loc[i, "times"]]:
        pvs.append(i)

groundpoints                    = [sf.earth(sf.eme2000).transform(pv.position, sf.eme2000, pv.date) for pv in pvs]
pos_info["40697"].loc[:, "lat"] = np.degrees([gp.latitude for gp in groundpoints])
pos_info["40697"].loc[:, "lon"] = np.degrees([gp.longitude for gp in groundpoints])

fig = go.Figure(data=go.Scattergeo(
    lat = [],
    lon = []
    )).update_layout(geo = dict(
                        projection_type = "mercator",
                        ))

fps = int(1440/res)
p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(fps), '-i', '-', '-vcodec', 'mpeg4', '-qscale', '5', 
           '-r', str(fps), 'video_short.avi', "-fflags", "+genpts"], stdin=PIPE)

for i in pos_info["40697"].index:
    fig.update_traces(
        lat = [pos_info["40697"].loc[i, "lat"]],
        lon = [pos_info["40697"].loc[i, "lon"]]
        )
    fig.update_layout(
        title = str(pos_info["40697"].loc[i, "time"])
        )
    
    img = Image.open(io.BytesIO(fig.to_image(format="png"))).convert('RGB')
    img.save(p.stdin, 'PNG')

p.stdin.close()
p.wait()