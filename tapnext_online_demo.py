import torch
import numpy as np
import imageio

from tapnet.tapnext.tapnext_online import TAPNextOnline
from tapnext_update_demo_full import get_queries
from visualize import visualize_and_save

def read_video(video_path):
    reader = imageio.get_reader(video_path)
    frames = []
    for i, im in enumerate(reader):
        frames.append(np.array(im))
    video = np.stack(frames)
    video = torch.from_numpy(video).float().cuda()  # (T, 720, 1920, 3)
    
    print(f"{video.shape[0]} frames in video")
    
    return video

if __name__ == "__main__":
    # Load the model
    model_path = "./checkpoints/bootstapnext_ckpt.npz"
    model = TAPNextOnline(model_path=model_path, resolution=(256, 256), radius=8, threshold=0.5, use_certainty=True)

    # Load the video frames
    # frames = np.load("frames.npy")
    # frames = torch.from_numpy(frames).cuda().float()  # Shape: (T, H, W, C)
    frames = read_video("/local/home/apadmanabhan/mt/MH_04_difficult.mp4")
    frames = frames[1750:]

    # Load the points and occlusion data
    # points = np.load("points.npy")
    # occluded = np.load("occluded.npy")

    # Run the model on each frame
    pred_tracks = []
    for t in range(frames.shape[0]):
        frame = frames[t:t+1]  # Shape: (1, H, W, C)
        queries, removed_indices = get_queries(t) 
        
        tracks, track_status = model(frame=frame, queries=queries, removed_indices=removed_indices)
        pred_tracks.append(tracks)

    pred_tracks = torch.cat(pred_tracks, dim=1)  # Shape: (1, T, N, 2)
    print(pred_tracks.shape)
    visualize_and_save(
        video_tensor=frames.to(dtype=torch.uint8).unsqueeze(0),
        trajectory_tensor=pred_tracks.permute((0,2,1,3)), 
        output_dir='/local/home/apadmanabhan/mt/tapnet/output/MH_04',
        video_filename='MH_04.mp4',
        fps=20,
    )