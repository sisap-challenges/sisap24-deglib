#!/usr/bin/env python3

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate recall for two h5 files')
    parser.add_argument('ground_truth', type=Path, help='The ground truth h5 file')
    parser.add_argument('prediction', type=Path, help='The prediction h5 file')
    parser.add_argument('-k', type=int, default=30, help='Number of results per query')

    return parser.parse_args()


def main():
    args = parse_args()
    gt_knns = load_h5(args.ground_truth)
    pred_knns = load_h5(args.prediction)

    recall = calc_recall(gt_knns, pred_knns, args.k)
    print('recall:', recall)


def load_h5(path: Path) -> np.ndarray:
    with h5py.File(path, 'r') as f:
        return f['knns'][()]


def calc_recall(ground_truth: np.ndarray, prediction: np.ndarray, k: int):
    assert prediction.shape[1] == k

    ground_truth = ground_truth[:, :k]  # only use first k ground truth entries

    correct = 0
    total = 0
    for gt, result in zip(ground_truth, prediction):
        correct += len(np.intersect1d(gt, result))
        total += k

    return correct / total


if __name__ == '__main__':
    main()
