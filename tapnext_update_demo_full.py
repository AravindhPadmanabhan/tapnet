import torch
import torch.nn.functional as F
import tqdm
import numpy as np

from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint, tracker_certainty
import mediapy as media

from visualize import visualize_and_save

def get_queries(t):

  queries1 = torch.tensor([[[  0.0000,  400.0, 200.0],
                            [  0.0000,  500.0, 250.0]]]).cuda()

  queries2 = torch.tensor([[[  0.0000, 400.0, 200.0],
                            [  25.0000, 300.0, 250.0]]]).cuda()

  removed_indices1 = []
  removed_indices2 = [1]

  if t == 26:
    return queries2, removed_indices2
  elif t < 26:
    return queries1, removed_indices1
  elif t > 26:
    return queries2, removed_indices1


def run_eval_per_frame(
    model,
    video,
    radius=8,
    threshold=0.5,
    use_certainty=False,
):
  B, T, H, W, C = video.shape
  # video = F.interpolate(video, (256,256), mode="bilinear", align_corners=True)

  # video = video.reshape(B * T, C, H, W)
  # video = F.interpolate(video.to(dtype=torch.float32), (256,256), mode="bilinear", align_corners=True, antialias=True)
  # video = video.reshape(B, T, 256, 256, 3)
  video = media.resize_video(video.squeeze(0).cpu().numpy(), (256, 256))
  video = torch.from_numpy(video).cuda().float().unsqueeze(0)

  video = (video/255.0) * 2.0 - 1.0
  # print(video)
  
  with torch.no_grad():
    queries, removed_indices = get_queries(0)
    queries = queries[:,:,[0,2,1]]  # t, y, x
    queries[:, :, 1:] = queries[:, :, 1:] * queries.new_tensor([(255)/(H-1), (255)/(W-1)])
    pred_tracks, track_logits, visible_logits = model(
        video=video[:, :1], query_points=queries
    )
    pred_visible = visible_logits > 0
    pred_tracks, pred_visible = [pred_tracks.cpu()], [pred_visible.cpu()]
    pred_track_logits, pred_visible_logits = [track_logits.cpu()], [
        visible_logits.cpu()
    ]
    for frame in tqdm.tqdm(range(1, video.shape[1])):
      # ***************************************************
      # HERE WE RUN POINT TRACKING IN PURELY ONLINE FASHION
      # ***************************************************
      queries, removed_indices = get_queries(frame)
      queries = queries[:,:,[0,2,1]]  # t, y, x
      queries[:, :, 1:] = queries[:, :, 1:] * queries.new_tensor([(255)/(H-1), (255)/(W-1)])
      (
          curr_tracks,
          curr_track_logits,
          curr_visible_logits,
      ) = model(
          video=video[:, frame : frame + 1],
          query_points=queries,
          removed_indices=removed_indices,
      )

      curr_visible = curr_visible_logits > 0
      # ***************************************************
      pred_tracks.append(curr_tracks.cpu())
      pred_visible.append(curr_visible.cpu())
      pred_track_logits.append(curr_track_logits.cpu())
      pred_visible_logits.append(curr_visible_logits.cpu())


    tracks = torch.cat(pred_tracks, dim=1).transpose(1, 2)
    pred_visible = torch.cat(pred_visible, dim=1).transpose(1, 2)
    track_logits = torch.cat(pred_track_logits, dim=1).transpose(1, 2)
    visible_logits = torch.cat(pred_visible_logits, dim=1).transpose(1, 2)


    pred_certainty = tracker_certainty(tracks, track_logits, radius)
    pred_visible_and_certain = (
        F.sigmoid(visible_logits) * pred_certainty
    ) > threshold

    if use_certainty:
      occluded = ~(pred_visible_and_certain.squeeze(-1))
    else:
      occluded = ~(pred_visible.squeeze(-1))

  tracks = tracks.permute(0,2,1,3)
  tracks = tracks * tracks.new_tensor([(H-1) / (255), (W-1) / (255)])
  tracks = tracks[...,[1,0]]

  occluded = occluded.permute(0,2,1)  

  return tracks, occluded

if __name__ == "__main__":
  # Load the dataset
  video = np.load("frames.npy")
  video = torch.from_numpy(video).cuda()
  video = video.unsqueeze(0)

  # I am either multiplying y with W and x with H in the tracks
  # Or I should just use media.resize instead of F.interpolate

  # Load the model
  model = TAPNext(image_size=(256, 256))
  ckpt_path = './checkpoints/bootstapnext_ckpt.npz'
  model = restore_model_from_jax_checkpoint(model, ckpt_path)
  model.cuda()

  # Inference
  with torch.amp.autocast('cuda', dtype=torch.float16, enabled=True):
    tracks, occluded = run_eval_per_frame(model, video, use_certainty=False)

  # print(tracks)
  
  visualize_and_save(video, tracks.permute(0,2,1,3), output_dir='/local/home/apadmanabhan/mt/tapnet/output', video_filename='tracked_video.mp4', fps=30)

