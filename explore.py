import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DataProcessing import *
import time
import plotly.graph_objects as go


voxel_train, seg_train, train_batch_size = data_read('data/train_val/candidate{}.npz', 584, notebook=False)
voxel_test, seg_test, test_batch_size = data_read('data/test/candidate{}.npz', 584, notebook=False)
train_label = pd.read_csv('data/train_val.csv').values[:, 1].astype(int)

volume = voxel_train[83]
r = 100
c = 100
print('data ready')
# Define frames
nb_frames = 100

fig = go.Figure(frames=[go.Frame(data=go.Surface(
    z=(10 - k * 0.1) * np.ones((r, c)),
    surfacecolor=np.flipud(volume[99 - k]),
    cmin=0, cmax=200
    ),
    name=str(k) # you need to name the frame for the animation to behave properly
    )
    for k in range(nb_frames)])
print('step 1')
# Add data to be displayed before animation starts
fig.add_trace(go.Surface(
    z=10 * np.ones((r, c)),
    surfacecolor=np.flipud(volume[99]),
    colorscale='Gray',
    cmin=0, cmax=200,
    colorbar=dict(thickness=20, ticklen=4)
    ))
print('step 2')


def frame_args(duration):
    return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

sliders = [
            {
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], frame_args(0)],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(fig.frames)
                ],
            }
        ]

# Layout
fig.update_layout(
         title='Slices in volumetric data',
         width=600,
         height=600,
         scene=dict(
                    zaxis=dict(range=[-0.1, 10], autorange=False),
                    aspectratio=dict(x=1, y=1, z=1),
                    ),
         updatemenus = [
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;", # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;", # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
         ],
         sliders=sliders
)
print('step 3')
fig.show()
