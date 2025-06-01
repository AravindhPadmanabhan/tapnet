import os
import sys
import numpy as np
from tqdm import tqdm
import time
import datetime
import math

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from eval_utils import Evaluator, compute_tapvid_metrics, get_dataloaders
from log_utils import log_eval_metrics, get_args, print_args


from tapnet.tapnext.tapnext_online import TAPNextOnline
from randomize import generate_random_lifetimes

@torch.no_grad()
def evaluate(args, val_dataloader, verbose=False):

    evaluator = Evaluator()
    total_frames = 0
    total_time = 0

    for j, (video, trajectory, visibility, query_points_i) in enumerate(tqdm(val_dataloader, disable=verbose, file=sys.stdout)):
        # Timer start
        start_time = time.time()
        total_frames += video.shape[1]

        query_points_i = query_points_i.cuda(non_blocking=True)      # (1, N, 3)
        trajectory = trajectory.cuda(non_blocking=True)              # (1, T, N, 2)
        visibility = visibility.cuda(non_blocking=True)              # (1, T, N)
        video = video.cuda(non_blocking=True)                    # (1, T, 3, H, W)
        B, T, N, _ = trajectory.shape
        _, _, _, H, W = video.shape
        device = video.device

        model =  TAPNextOnline(model_path=args.checkpoint_path, resolution=(256, 256), radius=8, threshold=0.5, use_certainty=True, device=device)

        # Change (t, y, x) to (t, x, y)
        queries = query_points_i.clone().float()
        queries = torch.stack([queries[:, :, 0], queries[:, :, 2], queries[:, :, 1]], dim=2).to(device)

        ids_list, removed_indices_list, new_queries_list, gt_traj, gt_vis, queries_sorted, end_frames = generate_random_lifetimes(
                    T,
                    queries.clone(),
                    trajectory.clone(),
                    visibility.clone(),
                    t=int(T/5),
                )
        traj = torch.zeros_like(trajectory).to(device)  # (1, T, N, 2)
        vis = torch.zeros_like(visibility).to(device)  # (1, T, N)

        is_first_step = True
        delay = 0
        init_img = video[:, 0, :, :, :] # (1, 3, H, W)
        for i in range(1, T):
            queries_in = queries_sorted[:,ids_list[i]].squeeze(0)
            queries_in[:,0] -= delay
            if is_first_step:
                if queries_in.shape[0] == 0:
                    init_img = video[:, i, :, :, :] # (1, 3, H, W)
                    delay += 1
                    continue
                __ = model(frame=init_img.permute(0,2,3,1), queries=queries_in.unsqueeze(0))
                is_first_step = False
            
            if queries_in.shape[0] == 0:
                break
            pred_traj, pred_vis = model(frame=video[:,i].permute(0,2,3,1), queries=queries_in.unsqueeze(0), removed_indices=removed_indices_list[i])
            traj[0, i, ids_list[i]] = pred_traj[0,0]
            vis[0, i, ids_list[i]] = pred_vis[0,0]

        # Timer end
        total_time += time.time() - start_time

        # === === ===
        # From CoTracker
        query_points = queries_sorted.clone().cpu().numpy()
        gt_tracks = gt_traj.permute(0, 2, 1, 3).cpu().numpy()
        gt_occluded = torch.logical_not(gt_vis.clone().permute(0, 2, 1)).cpu().numpy()
        pred_occluded = torch.logical_not(vis.clone().permute(0, 2, 1)).cpu().numpy()
        pred_tracks = traj.permute(0, 2, 1, 3).cpu().numpy()
        # === === ===


        out_metrics = compute_tapvid_metrics(query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks, "first", end_frames=end_frames.cpu().numpy())
        if verbose:
            print(f"Video {j}/{len(val_dataloader)}: AJ: {out_metrics['average_jaccard'][0] * 100:.2f}, delta_avg: {out_metrics['average_pts_within_thresh'][0] * 100:.2f}, OA: {out_metrics['occlusion_accuracy'][0] * 100:.2f}", flush=True)
        evaluator.update(out_metrics)
        
    fps = total_frames / total_time
    print(f"Evaluation FPS: {fps:.2f}", flush=True)

    results = evaluator.get_results()
    smaller_delta_avg = results["delta_avg"]
    aj = results["aj"]
    oa = results["oa"]

    
    log_eval_metrics(results)

    return smaller_delta_avg, aj, oa


def main_worker(args):
    # init_distributed_mode(args)
    # fix_random_seeds(args.seed)

    print_args(args)
    start_time = time.time()

    # ##### Data #####
    _, val_dataloader = get_dataloaders(args)
    # ##### ##### #####

    delta, aj, oa = evaluate(args, val_dataloader, verbose=True)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print()
    print('Validation time {}'.format(total_time_str))

    return delta, aj, oa

if __name__ == '__main__':
    args = get_args()
    deltas = []
    aj_scores = []
    oa_scores = []
    for i in range(1):
        delta, aj, oa = main_worker(args)
        deltas.append(delta)
        aj_scores.append(aj)
        oa_scores.append(oa)
    print(f"Average Delta: {np.mean(deltas):.3f} ± {np.std(deltas):.3f}")
    print(f"Average AJ: {np.mean(aj_scores):.3f} ± {np.std(aj_scores):.3f}")
    print(f"Average OA: {np.mean(oa_scores):.3f} ± {np.std(oa_scores):.3f}")
    print("Done")