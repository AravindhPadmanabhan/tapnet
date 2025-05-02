import pickle
import numpy as np

# Script to save a TAPVID video as a numpy file (to be used in tapnext_online_demo.py)
with open("./assets/tapvid_davis/tapvid_davis.pkl", "rb") as f:
    davis_data = pickle.load(f)

first_video_name = list(davis_data.keys())[0]
video_data = davis_data[first_video_name]

frames = video_data['video']
points = video_data['points']
occluded = video_data['occluded']

np.save("frames.npy", frames)
np.save("points.npy", points)
np.save("occluded.npy", occluded)
