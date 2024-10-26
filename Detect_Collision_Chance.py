import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta, UTC

prop_data = pd.read_json('Propagation_Data.json')
payload_data = prop_data[prop_data['object_type'] == 'PAYLOAD']

with open('Metadata.json') as file:
    datetimes = json.load(file)['datetimes']
dt_start    = datetime.strptime(datetimes['start'], '%Y-%m-%dT%H:%M:%S.%fZ')
dt_timestep = datetimes['timestep']
dt_steps    = datetimes['steps']
dts = [(dt_start + timedelta(seconds=dt_timestep * t)) for t in range(0, dt_steps)]

# TODO: Check for objects that are within threshold
'''
xs = prop_data['x'].apply(lambda x: list(map(lambda a: a//threshold, x)))
ys = prop_data['y'].apply(lambda x: list(map(lambda a: a//threshold, x)))
zs = prop_data['z'].apply(lambda x: list(map(lambda a: a//threshold, x)))

prop_data['quad'] = [[quad for quad in zip(xs[sat], ys[sat], zs[sat])] for sat in prop_data.index]
'''

for step, dt in enumerate(dts):
    max_vel = np.array([max([vel_x[step] for vel_x in payload_data['vel_x']]),
                        max([vel_y[step] for vel_y in payload_data['vel_y']]),
                        max([vel_z[step] for vel_z in payload_data['vel_z']])
                        ])
    d_pos = max_vel * dt_timestep

    payload_data['quad_x'] = [x[step]//d_pos[0] for x in payload_data['pos_x']]
    payload_data['quad_y'] = [y[step]//d_pos[1] for y in payload_data['pos_y']]
    payload_data['quad_z'] = [z[step]//d_pos[2] for z in payload_data['pos_z']]

    qx = payload_data['quad_x'].sort_values()
    pot_col = []

    i = 0
    lx = len(qx)
    while i < lx:
        i0 = i
        i+=1
        while qx.iloc[i] - qx.iloc[i0] <= 1 and i < lx:
            i+=1
        if i - i0 > 1:
            y_check = qx.iloc[i0:i].index
            qy = payload_data.loc[y_check, 'quad_y'].sort_values()

            j = 0
            ly = len(qy)
            while j < ly:
                j0 = j
                j+=1
                while qy.iloc[j] - qy.iloc[j0] <= 1 and j < ly:
                    j+=1
                if j - j0 > 1:
                    z_check = qy.iloc[j0:j].index
                    qz = payload_data.loc[z_check, 'quad_z'].sort_values()

                    k = 0
                    lz = len(qz)
                    while k < lz:
                        k0 = k
                        k+=1
                        while qz.iloc[k] - qz.iloc[k0] <=1 and k < lz:
                            k+=1
                        if k - k0 > 1:
                            pot_col.append(qz.iloc[k0:k].index)

print(pot_col)