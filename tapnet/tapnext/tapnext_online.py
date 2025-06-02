import torch
from torch import nn
from mediapy import resize_video
import torch.nn.functional as F
from typing import List

from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint, tracker_certainty

class TAPNextOnline(nn.Module):
    def __init__(self, model_path, resolution=(256,256), radius=8, threshold=0.5, use_certainty=True, device='cuda'):
        super(TAPNextOnline, self).__init__()
        self.device = device
        self.resolution = resolution
        self.radius = radius
        self.threshold = threshold
        self.use_certainty = use_certainty

        self.model = TAPNext(image_size=(256, 256))
        self.model = restore_model_from_jax_checkpoint(self.model, model_path)
        self.model = self.model.to(device)
        self.model.eval()

    def reset(self):
        """Reset the model state if needed."""
        self.model.state = None

    @torch.inference_mode()
    def forward(self, frame: torch.Tensor, queries: torch.Tensor, removed_indices: List[int] = []):
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=True):
            T, H, W, C = frame.shape
            assert T == 1, "Only one frame is supported for online tracking"
            frame = resize_video(frame.to(dtype=torch.uint8).cpu().numpy(), self.resolution)  # 1, 256, 256, 3
            frame = torch.from_numpy(frame).cuda().float().unsqueeze(0)  # 1, 1, 256, 256, 3
            frame = (frame/255.0) * 2.0 - 1.0

            queries = queries[:,:,[0,2,1]]  # t, y, x
            queries[:, :, 1:] = queries[:, :, 1:] * queries.new_tensor([(self.resolution[0]-1)/(H-1), (self.resolution[1]-1)/(W-1)])
            queries = queries.to(self.device).float()

            tracks, track_logits, visible_logits = self.model(video=frame, query_points=queries, removed_indices=removed_indices)
            pred_visible = (visible_logits > 0).transpose(1,2)

            pred_certainty = tracker_certainty(tracks.transpose(1,2), track_logits.transpose(1,2), radius=self.radius)
            pred_visible_and_certain = (torch.sigmoid(visible_logits.transpose(1,2)) * pred_certainty) > self.threshold

            if self.use_certainty:
              track_status = pred_visible_and_certain.squeeze(-1)
            else:
              track_status = pred_visible.squeeze(-1)

            tracks = tracks * tracks.new_tensor([(H-1) / (255), (W-1) / (255)])
            tracks = tracks[...,[1,0]]

            track_status = track_status.permute(0,2,1)

        return tracks, track_status
