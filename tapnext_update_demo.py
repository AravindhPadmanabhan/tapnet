import torch
import torchvision
import torch.nn.functional as F
import tqdm
import numpy as np

from tapnet import evaluation_datasets
from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint, tracker_certainty
from torch_tapnext_demo import deterministic_eval

from visualize import visualize_and_save

def get_queries(t):

  queries1 = torch.tensor([[[  0.0000,  50.0, 150.0],
                            [  0.0000,  70.0, 120.0]]]).cuda()

  queries2 = torch.tensor([[[  0.0000, 50.0, 150.0],
                            [  15.0000, 30.0, 130.0]]]).cuda()

  removed_indices1 = []
  removed_indices2 = [1]

  if t == 16:
    return queries2, removed_indices2
  elif t < 16:
    return queries1, removed_indices1
  elif t > 16:
    return queries2, removed_indices1


def run_eval_per_frame(
    model,
    batch,

    radius=8,
    threshold=0.5,
    use_certainty=False,
):
  with torch.no_grad():
    queries, removed_indices = get_queries(0)
    pred_tracks, track_logits, visible_logits = model(
        video=batch['video'][:, :1], query_points=queries
    )
    pred_visible = visible_logits > 0
    pred_tracks, pred_visible = [pred_tracks.cpu()], [pred_visible.cpu()]
    pred_track_logits, pred_visible_logits = [track_logits.cpu()], [
        visible_logits.cpu()
    ]
    for frame in tqdm.tqdm(range(1, batch['video'].shape[1])):
      # ***************************************************
      # HERE WE RUN POINT TRACKING IN PURELY ONLINE FASHION
      # ***************************************************
      queries, removed_indices = get_queries(frame)
      (
          curr_tracks,
          curr_track_logits,
          curr_visible_logits,
      ) = model(
          video=batch['video'][:, frame : frame + 1],
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
        torch.sigmoid(visible_logits) * pred_certainty
    ) > threshold

    if use_certainty:
      occluded = ~(pred_visible_and_certain.squeeze(-1))
    else:
      occluded = ~(pred_visible.squeeze(-1))

  return (tracks.numpy()[..., ::-1], occluded,)

if __name__ == "__main__":
  # Load the dataset
  davis_dataset = evaluation_datasets.create_davis_dataset(
      davis_points_path='./assets/tapvid_davis/tapvid_davis.pkl',
      query_mode='first',
      full_resolution=False,
      resolution=(256, 256),
  )

  cached_dataset = []
  for j, batch in enumerate(davis_dataset):
    cached_dataset.append(batch)

  # Load the model
  model = TAPNext(image_size=(256, 256))
  ckpt_path = './checkpoints/bootstapnext_ckpt.npz'
  model = restore_model_from_jax_checkpoint(model, ckpt_path)
  model.cuda()

  # Inference
  standard_eval_scalars_list = []
  preds = []
  video_tensor = None
  for batch in deterministic_eval(cached_dataset):
    video_tensor = batch['video']
    batch = {k: torch.from_numpy(v).cuda().float() for k, v in batch.items()}
    with torch.amp.autocast('cuda', dtype=torch.float16, enabled=True):
      tracks, occluded = run_eval_per_frame(
          model, batch, use_certainty=False
      )
    # standard_eval_scalars_list.append(scores)
    preds.append((tracks, occluded))
    break

  print(tracks)

  visualize_and_save(batch['video'], torch.from_numpy(tracks.copy()), output_dir='/local/home/apadmanabhan/mt/tapnet/output', video_filename='tracked_video.mp4', fps=30)

