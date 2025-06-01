import os
import argparse
from pathlib import Path
import torch

def log_eval_metrics(results):

    print(f"delta_avg: {results['delta_avg']:.2f}")
    print(f"delta_1: {results['delta_1']:.2f}")
    print(f"delta_2: {results['delta_2']:.2f}")
    print(f"delta_4: {results['delta_4']:.2f}")
    print(f"delta_8: {results['delta_8']:.2f}")
    print(f"delta_16: {results['delta_16']:.2f}")
    print(f"AJ: {results['aj']:.2f}")
    print(f"OA: {results['oa']:.2f}")

def print_args(args):
    print("====== Validation ======")
    print(f"Evaluating on: {args.eval_dataset}")
    print(f"Checkpoint Path: {args.checkpoint_path}")
    print("====== ======= ======\n")

def get_args():
    parser = argparse.ArgumentParser("Track-On")

    # === Data Related Parameters ===
    parser.add_argument('--tapvid_root', type=str, default=None)
    parser.add_argument('--eval_dataset', type=str, choices=["davis", "rgb_stacking", "kinetics", "robotap"], default="davis")
    parser.add_argument('--augmentation', action="store_true")
    parser.add_argument('--input_size', type=int, nargs=2, default=[384, 512])

    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

    return args