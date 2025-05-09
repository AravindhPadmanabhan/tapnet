import torch
import numpy as np

from tapnet.tapnext.tapnext_online import TAPNextOnline
from tapnext_update_demo_full import get_queries
from visualize import visualize_and_save

if __name__ == "__main__":
    # Load the model
    model_path = "./checkpoints/bootstapnext_ckpt.npz"
    model = TAPNextOnline(model_path=model_path, resolution=(256, 256), radius=8, threshold=0.5, use_certainty=True)

    # Load the video frames
    frames = np.load("frames.npy")
    frames = torch.from_numpy(frames).cuda().float()  # Shape: (T, H, W, C)
    # frames = frames.permute(0, 3, 1, 2)  # Shape: (T, C, H, W)

    # Load the points and occlusion data
    # points = np.load("points.npy")
    # occluded = np.load("occluded.npy")

    # Run the model on each frame
    pred_tracks = []
    conv1d_cache_log = []
    rglru_cache_log = []
    for t in range(frames.shape[0]):
        frame = frames[t:t+1]  # Shape: (1, H, W, C)
        queries = torch.tensor([[[200.0, 400.0, 300.0],
                                 [200.0, 100.0, 100.0]]], device='cuda')  # Example query
        removed_indices = []  # Example removed indices
        
        tracks, track_status = model(frame=frame, queries=queries, removed_indices=removed_indices)

        conv1d_cache = []
        rglru_cache = []
        for i in range(len(model.model.state.hidden_state)):
          conv1d_cache.append(model.model.state.hidden_state[i].conv1d_state)  # 1025, 3, 768
          rglru_cache.append(model.model.state.hidden_state[i].rg_lru_state)  # 1025, 768
        conv1d_cache_log.append(torch.stack(conv1d_cache, dim=0))
        rglru_cache_log.append(torch.stack(rglru_cache, dim=0))

        pred_tracks.append(tracks)

    # full_conv1d_cache = torch.stack(conv1d_cache_log, dim=0)  # Shape: (T, 12, 1025, 3, 768)
    # full_rglru_cache = torch.stack(rglru_cache_log, dim=0)  # Shape: (T, 12, 1025, 768)
    # np.save("conv1d_cache2.npy", full_conv1d_cache.cpu().numpy())
    # np.save("rglru_cache2.npy", full_rglru_cache.cpu().numpy())

    pred_tracks = torch.cat(pred_tracks, dim=1)  # Shape: (1, T, N, 2)
    print(pred_tracks)
    # print(((full_conv1d_cache[:,:,-1,:,:] - full_conv1d_cache[:,:,-2,:,:]) == 0.0).all())
    # np.save("pred_tracks2.npy", pred_tracks.cpu().numpy())
    