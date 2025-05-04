import torch
import torchvision
import torch.nn.functional as F
import tqdm
import numpy as np

from tapnet import evaluation_datasets
from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint, tracker_certainty

from visualize import visualize_and_save

# queries = [[[  0.0000,  87.8248, 131.9137],
#             [  0.0000,  78.0146, 139.7970],
#             [  0.0000,  69.6058, 153.4614],
#             [  0.0000, 100.4380, 117.7238],
#             [  0.0000, 107.9124, 149.2570]]]

queries = [[[  0.0000, 50.0, 200.0],
            [  8.0000, 30.0, 130.0]]]
            # [  0.0000,  70.0, 150.0]]]
            # [  20.0000, 100.4380, 117.7238],
            # [  20.0000, 107.9124, 149.2570]]]
queries = torch.tensor(queries).cuda()

conv1d_cache_log = []
rglru_cache_log = []

def run_eval_per_frame(
    model,
    batch,
    get_trackwise_metrics=True,
    radius=8,
    threshold=0.5,
    use_certainty=False,
):
  with torch.no_grad():
    print(batch['video'].shape)
    pred_tracks, track_logits, visible_logits, tracking_state = model(
        video=batch['video'][:, :1], query_points=queries
    )
    # print("Queries: ", batch['query_points'])
    pred_visible = visible_logits > 0
    pred_tracks, pred_visible = [pred_tracks.cpu()], [pred_visible.cpu()]
    pred_track_logits, pred_visible_logits = [track_logits.cpu()], [
        visible_logits.cpu()
    ]
    for frame in tqdm.tqdm(range(1, batch['video'].shape[1])):
      # ***************************************************
      # HERE WE RUN POINT TRACKING IN PURELY ONLINE FASHION
      # ***************************************************
      
    #   if frame == 8:
    #     for i in range(len(tracking_state.hidden_state)):
    #         tracking_state.hidden_state[i].conv1d_state[-1] = 0.0
    #         tracking_state.hidden_state[i].rg_lru_state[-1] = 0.0
      
      (
          curr_tracks,
          curr_track_logits,
          curr_visible_logits,
          tracking_state,
      ) = model(
          video=batch['video'][:, frame : frame + 1],
          state=tracking_state,
      )

      conv1d_cache = []
      rglru_cache = []
      if frame < 10:
        for i in range(len(tracking_state.hidden_state)):
          conv1d_cache.append(tracking_state.hidden_state[i].conv1d_state)
          rglru_cache.append(tracking_state.hidden_state[i].rg_lru_state)
        conv1d_cache_log.append(torch.stack(conv1d_cache, dim=0))
        rglru_cache_log.append(torch.stack(rglru_cache, dim=0))
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

    # conv1d_cache_check = []
    # rglru_cache_check = []
    # for i in range(len(conv1d_cache_log)):
    #   conv1d_cache_check.append((conv1d_cache_log[i][:,-3,:,:] == conv1d_cache_log[i][:,-2,:,:]).all().cpu().item())
    #   rglru_cache_check.append((rglru_cache_log[i][:,-3,:] == rglru_cache_log[i][:,-2,:]).all().cpu().item())

    # print("Conv1D cache check: ", conv1d_cache_check)
    # print("RG-LRU cache check: ", rglru_cache_check)
    # print("Cache same for same query at diff timestamps before query frame?: ", (conv1d_cache_log[1][:,-3,:,:] == conv1d_cache_log[2][:,-3,:,:]).all().cpu().item())
    # print("Cache same at same timestamp for diff queries before query frame?: ", (conv1d_cache_log[3][:,-3,:,:] == conv1d_cache_log[3][:,-2,:,:]).all().cpu().item())

    pred_certainty = tracker_certainty(tracks, track_logits, radius)
    pred_visible_and_certain = (
        torch.sigmoid(visible_logits) * pred_certainty
    ) > threshold

    if use_certainty:
      occluded = ~(pred_visible_and_certain.squeeze(-1))
    else:
      occluded = ~(pred_visible.squeeze(-1))

#   scalars = evaluation_datasets.compute_tapvid_metrics(
#       batch['query_points'].cpu().numpy(),
#       batch['occluded'].cpu().numpy(),
#       batch['target_points'].cpu().numpy(),
#       occluded.numpy() + 0.0,
#       tracks.numpy()[..., ::-1],
#       query_mode='first',
#       get_trackwise_metrics=get_trackwise_metrics,
#   )

  np.save('conv1d_cache_log.npy', conv1d_cache_log[6][:,-2:,:,:].cpu().numpy())
  np.save('rglru_cache_log.npy', rglru_cache_log[6][:,-2:,:].cpu().numpy())
  return (
      tracks.numpy()[..., ::-1],
      occluded,
    #   {k: v.sum(0) for k, v in scalars.items()},
  )


# @title Function for raw data to the input format {form-width: "25%"}
def deterministic_eval(cached_dataset, strided=False):
  if not strided:
    for sample in tqdm.tqdm(cached_dataset, disable=True):
      batch = sample['davis'].copy()
      # batch['video'] = (batch['video'] + 1) / 2
      batch['visible'] = np.logical_not(batch['occluded'])[..., None]
      batch['padding'] = np.ones(
          batch['query_points'].shape[:2], dtype=np.bool_
      )
      batch['loss_mask'] = np.ones(
          batch['target_points'].shape[:3] + (1,), dtype=np.float32
      )
      batch['appearance'] = np.ones(
          batch['target_points'].shape[:3] + (1,), dtype=np.float32
      )

      yield batch
  else:
    for sample in tqdm.tqdm(cached_dataset):
      batch = sample['davis'].copy()
      # batch['video'] = (batch['video'] + 1) / 2
      batch['visible'] = np.logical_not(batch['occluded'])[..., None]
      batch['padding'] = np.ones(
          batch['query_points'].shape[:2], dtype=np.bool_
      )
      batch['loss_mask'] = np.ones(
          batch['target_points'].shape[:3] + (1,), dtype=np.float32
      )
      batch['appearance'] = np.ones(
          batch['target_points'].shape[:3] + (1,), dtype=np.float32
      )
      backward_batch = {k: v.copy() for k, v in batch.items()}
      for key in ['visible', 'appearance', 'loss_mask', 'target_points']:
        backward_batch[key] = np.flip(backward_batch[key], axis=2)
      backward_batch['video'] = np.flip(backward_batch['video'], axis=1)
      backward_queries = (
          backward_batch['video'].shape[1]
          - backward_batch['query_points'][..., 0]
          - 1
      )
      backward_batch['query_points'][..., 0] = backward_queries
      yield batch, backward_batch

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
          model, batch, get_trackwise_metrics=False, use_certainty=False
      )
    # standard_eval_scalars_list.append(scores)
    preds.append((tracks, occluded))
    break


#   print('')
#   print(
#       'AJ',
#       np.mean([
#           standard_eval_scalars_list[k]['average_jaccard']
#           for k in range(len(standard_eval_scalars_list))
#       ]),
#   )
#   print(
#       'OA',
#       np.mean([
#           standard_eval_scalars_list[k]['occlusion_accuracy']
#           for k in range(len(standard_eval_scalars_list))
#       ]),
#   )
#   print(
#       'PTS',
#       np.mean([
#           standard_eval_scalars_list[k]['average_pts_within_thresh']
#           for k in range(len(standard_eval_scalars_list))
#       ]),
#   )

#   print("tracks: ", tracks)
#   np.save('tracks2.npy', tracks[:,1].copy())

#   visualize_and_save(batch['video'], torch.from_numpy(tracks.copy()), output_dir='/local/home/apadmanabhan/mt/tapnet/output', video_filename='tracked_video.mp4', fps=30)

