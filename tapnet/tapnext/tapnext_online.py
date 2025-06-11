import torch
from torch import nn
from mediapy import resize_video
import torch.nn.functional as F
from typing import List

from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint, tracker_certainty
from tapnet.tapnext.split_patches import PatchSplitter

class TAPNextOnline(nn.Module):
    def __init__(self, model_path, resolution=(256,256), radius=8, threshold=0.5, use_certainty=True, device='cuda'):
        super(TAPNextOnline, self).__init__()
        self.device = device
        self.resolution = resolution
        self.radius = radius
        self.threshold = threshold
        self.use_certainty = use_certainty
        if not use_certainty:
            self.threshold = 0.0

        self.model = TAPNext(image_size=(256, 256))
        self.model = restore_model_from_jax_checkpoint(self.model, model_path)
        self.model = self.model.to(device)
        self.model.eval()
        self.splitter = None

    def reset(self):
        """Reset the model state if needed."""
        self.model.state = None

    @torch.inference_mode()
    def forward(self, frame: torch.Tensor, queries: torch.Tensor, removed_indices: List[int] = []):
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=True):
            T, H, W, C = frame.shape
            assert T == 1, "Only one frame is supported for online tracking"
            if self.splitter is None:
                self.splitter = PatchSplitter(def_shape=self.resolution, original_image_dims=(H, W))
            frame = self.splitter.split_video(frame)
            frame = frame.unsqueeze(1) # 6, 1, 256, 256, 3
            frame = (frame/255.0) * 2.0 - 1.0

            queries = queries.to(self.device).float()
            removed_mask = self.splitter.split_removed_indices(removed_indices)    # 6, N
            t = self.model.state.step-1 if self.model.state is not None else 0
            queries = self.splitter.split_queries(queries, t=t)  # 6, N, 3
            queries = queries[:,:,[0,2,1]]  # t, y, x

            tracks, track_logits, visible_logits = self.model(video=frame, query_points=queries, removed_mask=removed_mask)
            print("track logits shape: ", track_logits.shape, "  ", track_logits.min(), "  ", track_logits.max())
            pred_visible = (visible_logits).transpose(1,2)

            pred_certainty = tracker_certainty(tracks.transpose(1,2), track_logits.transpose(1,2), radius=self.radius)
            pred_visible_and_certain = (torch.sigmoid(visible_logits.transpose(1,2)) * pred_certainty)
            print("pred_visible_and_certain: ", pred_visible_and_certain.shape, "  ", pred_visible_and_certain.min(), "  ", pred_visible_and_certain.max())
            if self.use_certainty:
              track_status = pred_visible_and_certain.squeeze(-1)

            else:
              track_status = pred_visible.squeeze(-1)

            tracks = tracks[...,[1,0]]
            track_status = track_status.permute(0,2,1)

            tracks, track_status = self.splitter.combine_tracks(tracks.squeeze(1), track_status.squeeze(1))  # CHECK SHAPE

        return tracks.unsqueeze(0), (track_status > self.threshold).unsqueeze(0)
