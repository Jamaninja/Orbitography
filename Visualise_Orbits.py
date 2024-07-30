import orekit
vm = orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir, absolutedate_to_datetime, datetime_to_absolutedate
setup_orekit_curdir()

from org.orekit.propagation.analytical.tle import TLE, TLEPropagator # type: ignore
from org.orekit.frames import Frame, Transform, StaticTransform # type: ignore
from Orbitography_Functions import PlotFunctions, SatelliteFunctions

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

import io 
from PIL import Image
import ffmpeg
import subprocess

sat_data = {
    "SENTINEL 2A": {
        "norad_id":     "40697",  # For Space-Track TLE queries
        "cospar_id":    "",  # For laser ranging data queries
        "sic_id":       "",  # For writing in CPF files
        "mass":         8000.0, # kg; TODO: compute proper value
        "cross_section":100.0, # m2; TODO: compute proper value
        "cd":           2.0, # TODO: compute proper value
        "cr":           1.0  # TODO: compute proper value
    }
}

with open("tle.txt") as file:
    tle = file.readlines()

tle1, tle2 = tle[::2], tle[1::2]
tle_info, pos_info = {}, {}

for i,_ in enumerate(tle1):
    tle = TLE(tle1[i],tle2[i])
    if (sat := str(tle).split("\n")[2].split(" ")[1]) not in tle_info:
        tle_info[sat] = pd.DataFrame(columns=["tle", "epoch", "start", "end", "times"])
        pos_info[sat] = pd.DataFrame(columns=["pv", "time", "lat", "lon"])
        n = i
        
        if sat == "40697":
            sat_data["SENTINEL 2A"]["TLE"] = (tle1[i],tle2[i])

    tle_info[sat].loc[i-n, "tle"]   = tle
    tle_info[sat].loc[i-n, "epoch"] = absolutedate_to_datetime(tle.getDate())
    tle_info[sat].loc[i-n, "times"] = []

sat_epoch = lambda ind: tle_info["40697"].loc[ind, "epoch"] if ind >=0 else tle_info["40697"].loc[tle_info["40697"].index[ind], "epoch"]

res         = 5 # minutes per frame
time_start  = sat_epoch(0)
time_end    = sat_epoch(-1)
time_start  = datetime(time_start.year, time_start.month, time_start.day, 0, 0, 0, 0)               # Sets start time to 00:00 UTC on the day of the first epoch
time_end    = datetime(time_end.year, time_end.month, time_end.day, 0, 0, 0, 0) + timedelta(days=1) # Sets end time to 00:00 UTC of the day after the last epoch
time_array  = [time_start + timedelta(minutes=m) for m in range(0, int((time_end-time_start).total_seconds()/60 + 1), res)]

prop_tle_info, prop_num_info = {}, {}
prop_tle_info["40697"] = pd.DataFrame(columns=["pv", "time", "lat", "lon"])
prop_num_info["40697"] = pd.DataFrame(columns=["pv", "time", "lat", "lon"])

prop_tle_info["40697"].loc[:, "time"] = time_array
prop_num_info["40697"].loc[:, "time"] = time_array

sf = SatelliteFunctions(sat_data=sat_data)

# TLE Propagation
prop_tle_info["40697"].loc[:, "pv"]     = sf.propagateTLE(sat="SENTINEL 2A", resolution=res*60, start=datetime_to_absolutedate(time_start), end=datetime_to_absolutedate(time_end))
groundpoints                            = [sf.earth(sf.eme2000).transform(pv.position, sf.eme2000, pv.date) for pv in prop_tle_info["40697"].loc[:, "pv"]]
prop_tle_info["40697"].loc[:, "lat"]    = np.degrees([gp.latitude for gp in groundpoints])
prop_tle_info["40697"].loc[:, "lon"]    = np.degrees([gp.longitude for gp in groundpoints])

# Numerical Propagation
prop_num_info["40697"].loc[:, "pv"]     = sf.propagateNumerical(sat="SENTINEL 2A", resolution=res*60, start=datetime_to_absolutedate(time_start), end=datetime_to_absolutedate(time_end))
groundpoints                            = [sf.earth(sf.eme2000).transform(pv.position, sf.eme2000, pv.date) for pv in prop_num_info["40697"].loc[:, "pv"]]
prop_num_info["40697"].loc[:, "lat"]    = np.degrees([gp.latitude for gp in groundpoints])
prop_num_info["40697"].loc[:, "lon"]    = np.degrees([gp.longitude for gp in groundpoints])

pos_info["40697"].loc[:, "time"] = time_array
pvs = []

for i in tle_info["40697"].index:
    if not i: # Assigns the start and end points for each epoch to propagate between, usually the midpoint between epochs
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
            tle_info["40697"].loc[i, "times"].append(t)     # Creates an array of time steps that lie between the start and end points of each epoch

        elif t > tle_info["40697"].loc[i, "end"]:
            break                                           # Breaks t loop once time step is beyond current range

        else:
            tle_info["40697"].loc[i, "times"].append(t)     # Added seperately to prevent a 1 in 3.6 billion odds crash
    
    time_array = [x for x in time_array if x not in tle_info["40697"].loc[i, "times"]]                                              # Removes time steps that have already been assigned
    propagator = TLEPropagator.selectExtrapolator(tle_info["40697"].loc[i, "tle"])                                                  # Creates a TLE propagation model for the epoch at index i
    for pv in [propagator.propagate(datetime_to_absolutedate(t)).getPVCoordinates() for t in tle_info["40697"].loc[i, "times"]]:    # Calculates PV coordinates for all time steps that use this epoch
        pvs.append(pv)

pos_info["40697"].loc[:, "pv"]  = pvs                                                                                               # And adds them to a dataframe
groundpoints                    = [sf.earth(sf.eme2000).transform(pv.position, sf.eme2000, pv.date) for pv in pos_info["40697"].loc[:, "pv"]]
pos_info["40697"].loc[:, "lat"] = np.degrees([gp.latitude for gp in groundpoints])
pos_info["40697"].loc[:, "lon"] = np.degrees([gp.longitude for gp in groundpoints])

fps = int(144/res)
#p = subprocess.Popen(["ffmpeg", "-y", "-f", "image2pipe", "-vcodec", "bmp", "-r", str(fps), "-i", "-", "-vcodec", "mpeg4", "-q:v", "5", "-r", str(fps), "video_short.avi"], stdin=subprocess.PIPE) 
    # Fancy FFmpeg code that I repurposed from Marwan Alsabbagh at
    # https://stackoverflow.com/questions/13294919/can-you-stream-images-to-ffmpeg-to-construct-a-video-instead-of-saving-them-t

threshold = int(30 / res)
opacity = [np.exp(-x) for x in np.linspace(0, np.log(64), threshold)]

"""for i in pos_info["40697"].index:
    h = max(0, i - threshold)
    
    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        lat     = [pos_info["40697"].loc[i, "lat"]],
        lon     = [pos_info["40697"].loc[i, "lon"]],
        marker  = dict(color="red",size=3),
        ))
    
    fig.add_trace(go.Scattergeo(
        lat     = [prop_tle_info["40697"].loc[i, "lat"]],
        lon     = [prop_tle_info["40697"].loc[i, "lon"]],
        marker  = dict(color="green",size=3),
        ))
    
    fig.add_trace(go.Scattergeo(
        lat     = [prop_num_info["40697"].loc[i, "lat"]],
        lon     = [prop_num_info["40697"].loc[i, "lon"]],
        marker  = dict(color="blue",size=3),
        ))
        
    for j in range(h,i):
        fig.add_trace(go.Scattergeo(
            lat     = pos_info["40697"].loc[j:j+1, "lat"],
            lon     = pos_info["40697"].loc[j:j+1, "lon"],
            mode    = "lines",
            line    = dict(color="red",width=1.5),
            opacity = opacity[i-j-1],
            ))
        
        fig.add_trace(go.Scattergeo(
            lat     = prop_tle_info["40697"].loc[j:j+1, "lat"],
            lon     = prop_tle_info["40697"].loc[j:j+1, "lon"],
            mode    = "lines",
            line    = dict(color="green",width=1.5),
            opacity = opacity[i-j-1],
            )) 
        
        fig.add_trace(go.Scattergeo(
            lat     = prop_num_info["40697"].loc[j:j+1, "lat"],
            lon     = prop_num_info["40697"].loc[j:j+1, "lon"],
            mode    = "lines",
            line    = dict(color="blue",width=1.5),
            opacity = opacity[i-j-1],
            )) 

    fig.update_geos(
        projection_type     = "orthographic",
        center              = dict(lat=0, lon=-90),
        projection_rotation = dict(lat=0, lon=-90)
        #center              = dict(lat=pos_info["40697"].loc[i, "lat"],
        #                           lon=pos_info["40697"].loc[i, "lon"]),
        #projection_rotation = dict(lat=pos_info["40697"].loc[i, "lat"],
        #                           lon=pos_info["40697"].loc[i, "lon"],
        )
    
    '''fig.update_layout(
        #title       = str(pos_info["40697"].loc[i, "time"]),
        #title_x     = 0.5,
        #title_font  = dict(size=30),
        showlegend  = False,
        margin      = {"t":0,"b":0,"l":0,"r":0},
        height      = 500,
        width       = 500
        )'''

    img = Image.open(io.BytesIO(fig.to_image(format="png")))
    img.save(p.stdin, "BMP")
"""
dp_tle, dp_num = [], []
for i in pos_info["40697"].index:
    trans = Transform(pos_info["40697"].loc[i, "pv"].date, pos_info["40697"].loc[i, "pv"]).toStaticTransform().getInverse()
    dp_tle.append(trans.transformPosition(prop_tle_info["40697"].loc[i,"pv"].position))
    dp_num.append(trans.transformPosition(prop_num_info["40697"].loc[i,"pv"].position))

from scipy.fft import fft, fftfreq

xs_tle = np.array([p.x for p in dp_tle[:]])
n = xs_tle.size
timestep = res/3600
x_fft = fft(xs_tle)
freqs = fftfreq(n, d=timestep)[:n//2]

idx = np.argmax(np.abs(x_fft))
freq = freqs[idx]
print(freq)

import matplotlib.pyplot as plt

plt.plot(freqs, 2.0/n * np.abs(x_fft[:n//2]))
plt.grid()
plt.show()

"""
ys_tle = [abs(p.y) for p in dp_tle[:]]
zs_tle = [abs(p.z) for p in dp_tle[:]]

xs_num = [abs(p.x) for p in dp_num[:]]
ys_num = [abs(p.y) for p in dp_num[:]]
zs_num = [abs(p.z) for p in dp_num[:]]

fig = make_subplots(rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0)

fig.add_trace(
    go.Scatter(
        x       = pos_info["40697"].loc[:,"time"],
        y       = xs_tle, 
        mode    = "lines",
        marker  = dict(color = "blue")
        ),
    row=1, col=1
    )
fig.add_trace(
    go.Scatter(
        x       = pos_info["40697"].loc[:,"time"],
        y       = ys_tle, 
        mode    = "lines",
        marker  = dict(color = "red"),
        ),
    row=1, col=1
    )
fig.add_trace(
    go.Scatter(
        x       = pos_info["40697"].loc[:,"time"],
        y       = zs_tle, 
        mode    = "lines",
        marker  = dict(color = "green")
        ),
    row=1, col=1
    )

fig.add_trace(
    go.Scatter(
        x       = pos_info["40697"].loc[:,"time"],
        y       = xs_num, 
        mode    = "lines",
        marker  = dict(color = "blue")
        ),
    row=2, col=1
    )
fig.add_trace(
    go.Scatter(
        x       = pos_info["40697"].loc[:,"time"],
        y       = ys_num, 
        mode    = "lines",
        marker  = dict(color = "red")
        ),
    row=2, col=1
    )
fig.add_trace(
    go.Scatter(
        x       = pos_info["40697"].loc[:,"time"],
        y       = zs_num, 
        mode    = "lines",
        marker  = dict(color = "green")
        ),
    row=2, col=1
    )

fig.update_layout(
    showlegend = False
)

######

fig = make_subplots(rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0)

fig.add_trace(
    go.Scatter(
        x       = freq,
        y       = x_fft, 
        #mode    = "lines",
        marker  = dict(color = "blue")
        ),
    row=1, col=1
    )


fig.show()

p.stdin.close()
p.wait()
#sub.run(["ffmpeg", "-i", "left_video.avi", "-i", "right_video.avi", "-filter_complex", "hstack", "output_video.avi"])
"""