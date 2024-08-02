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
    prediction_path: Path = args.prediction

    print(f'{"eps":<8} {"recall"}')
    if prediction_path.is_file():
        recall = calc_for_prediction(args.prediction, gt_knns, args.k)
        eps = float(prediction_path.name[10:-3])
        print(f'{eps:<8} {recall}')
    elif prediction_path.is_dir():
        eps_and_recalls = []
        for file in prediction_path.iterdir():
            if file.is_file() and file.suffix == '.h5':
                eps = file.name[10:-3]
                recall = calc_for_prediction(file, gt_knns, args.k)
                eps_and_recalls.append((eps, recall))
        eps_and_recalls.sort(key=lambda x: float(x[0]))
        for eps, recall in eps_and_recalls:
            print(f'{eps:<8} {recall:.4f}')


def calc_for_prediction(prediction, gt_knns: np.ndarray, k: int):
    pred_knns = load_h5(prediction)
    return calc_recall(gt_knns, pred_knns, k)


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
