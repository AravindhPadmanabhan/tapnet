import torch
import numpy as np
import cv2
import os
from tqdm import tqdm

def visualize_and_save(video_tensor, trajectory_tensor, output_dir='output', video_filename='tracked_video.mp4', fps=30):
    # Remove batch dimension
    video = video_tensor[0]  # (T, H, W, 3)
    trajectories = trajectory_tensor[0]  # (N, T, 2)

    T, H, W, _ = video.shape
    N = trajectories.shape[0]

    # Make sure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(os.path.join(output_dir, video_filename), fourcc, fps, (W, H))

    for t in tqdm(range(T), desc="Saving frames and video"):
        frame = video[t].cpu().numpy()

        # Ensure correct dtype
        if frame.dtype != np.uint8:
            frame = ((frame+1) * 255 / 2).clip(0, 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw tracked points
        for n in range(N):
            x, y = trajectories[n, t].cpu().numpy()
            x, y = int(round(x)), int(round(y))
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(frame, (x, y), radius=4, color=(0, 255, 0), thickness=-1)

        # Write frame to video
        video_writer.write(frame)

        # Save frame as image
        frame_filename = os.path.join(frames_dir, f'frame_{t:04d}.png')
        cv2.imwrite(frame_filename, frame)

    video_writer.release()
    print(f"Saved video to {os.path.join(output_dir, video_filename)}")
    print(f"Saved frames to {frames_dir}")

