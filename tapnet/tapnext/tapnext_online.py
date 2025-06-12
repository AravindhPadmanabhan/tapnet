import torch
from torch import nn
from mediapy import resize_video
import torch.nn.functional as F
from typing import List, Tuple, Optional

from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint, tracker_certainty

def get_points_on_a_grid(
    size: int,
    extent: Tuple[float, ...],
    center: Optional[Tuple[float, ...]] = None,
    device: Optional[torch.device] = torch.device("cpu"),
):
    r"""Get a grid of points covering a rectangular region

    `get_points_on_a_grid(size, extent)` generates a :attr:`size` by
    :attr:`size` grid fo points distributed to cover a rectangular area
    specified by `extent`.

    The `extent` is a pair of integer :math:`(H,W)` specifying the height
    and width of the rectangle.

    Optionally, the :attr:`center` can be specified as a pair :math:`(c_y,c_x)`
    specifying the vertical and horizontal center coordinates. The center
    defaults to the middle of the extent.

    Points are distributed uniformly within the rectangle leaving a margin
    :math:`m=W/64` from the border.

    It returns a :math:`(1, \text{size} \times \text{size}, 2)` tensor of
    points :math:`P_{ij}=(x_i, y_i)` where

    .. math::
        P_{ij} = \left(
             c_x + m -\frac{W}{2} + \frac{W - 2m}{\text{size} - 1}\, j,~
             c_y + m -\frac{H}{2} + \frac{H - 2m}{\text{size} - 1}\, i
        \right)

    Points are returned in row-major order.

    Args:
        size (int): grid size.
        extent (tuple): height and with of the grid extent.
        center (tuple, optional): grid center.
        device (str, optional): Defaults to `"cpu"`.

    Returns:
        Tensor: grid.
    """
    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]

    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)

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

        self.local_grid_size = 8  # Size of the local grid to add
        self.local_grid_extent = 48  # Extent of the local grid in pixels

    def reset(self):
        """Reset the model state if needed."""
        self.model.state = None

    def add_support_grid(self, queries): 
        B = 1  # Assuming batch size is always 1 in this context
        if self.local_grid_size > 0:
            aug_queries = []

            for i in range(queries.shape[1]):
                frame = queries[0, i, 0].item()
                grid_pts = get_points_on_a_grid(
                    self.local_grid_size, 
                    (self.local_grid_extent, self.local_grid_extent),
                    (queries[0, i, 2].item(), queries[0, i, 1].item()),  # ensure x, y order
                    device=self.device,
                )
                grid_pts = torch.cat([torch.ones_like(grid_pts[:, :, :1]) * frame, grid_pts], dim=2)
                grid_pts = grid_pts.repeat(B, 1, 1)

                # Append query first, then its support grid
                aug_queries.append(queries[:, i:i+1, :])  # Extract and keep the original shape
                aug_queries.append(grid_pts)

            # Concatenate along the second dimension to maintain the interleaved pattern
            queries = torch.cat(aug_queries, dim=1)

        return queries
    
    def augment_removed_indices(self, removed_indices):
        if self.local_grid_size > 0:
            aug_removed_indices = []
            for i in range(len(removed_indices)):
                aug_index = removed_indices[i]*(1 + self.local_grid_size**2)
                aug_removed_indices.append(aug_index)
                removed_grid_indices = list(range(aug_index + 1, aug_index + 1 + self.local_grid_size**2))
                aug_removed_indices += removed_grid_indices

            removed_indices = aug_removed_indices
        return removed_indices

    @torch.inference_mode()
    def forward(self, frame: torch.Tensor, queries: torch.Tensor, removed_indices: List[int] = []):
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=True):
            T, H, W, C = frame.shape
            assert T == 1, "Only one frame is supported for online tracking"
            frame = resize_video(frame.to(dtype=torch.uint8).cpu().numpy(), self.resolution)  # 1, 256, 256, 3
            frame = torch.from_numpy(frame).cuda().float().unsqueeze(0)  # 1, 1, 256, 256, 3
            frame = (frame/255.0) * 2.0 - 1.0

            queries = self.add_support_grid(queries)
            removed_indices = self.augment_removed_indices(removed_indices)
            queries = queries[:,:,[0,2,1]]  # t, y, x
            queries[:, :, 1:] = queries[:, :, 1:] * queries.new_tensor([(self.resolution[0]-1)/(H-1), (self.resolution[1]-1)/(W-1)])
            queries = queries.to(self.device).float()

            tracks, track_logits, visible_logits = self.model(video=frame, query_points=queries, removed_indices=removed_indices)

            if self.local_grid_size > 0:
                step_size = 1 + self.local_grid_size**2
                tracks = tracks[:,:,::step_size]
                track_logits = track_logits[:,:,::step_size]
                visible_logits = visible_logits[:,:,::step_size]

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
