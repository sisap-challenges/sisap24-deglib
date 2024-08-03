#!/usr/bin/env python3

import argparse
from pathlib import Path

import csv
import glob
import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate recall for two h5 files')
    parser.add_argument(
        '--dbsize',
        type=str,
        choices=["100K", "300K", "10M", "30M", "100M"],
        help='The database size to use'
    )
    parser.add_argument(
        '--ground-truth', '-gt',
        type=Path,
        default=Path('data/gold-standard-dbsize=300K--public-queries-2024-laion2B-en-clip768v2-n=10k.h5'),
        help='The ground truth h5 file'
    )
    parser.add_argument(
        '--results',
        type=str,
        default='results',
        help='The prediction h5 file'
    )
    parser.add_argument(
        '-k',
        type=int,
        default=30,
        help='Number of results per query'
    )
    parser.add_argument(
        "--csvfile",
        type=Path,
        default=Path('results/res.csv'),
        help='Name of the csv file'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    gt_knns = load_h5(args.ground_truth)

    columns = ["data", "size", "algo", "buildtime", "querytime", "params", "recall"]
    with open(args.csvfile, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        print(f'{"eps":<8} {"recall"}')
        for res in get_all_results(args.results + "/" + args.dbsize):
            try:
                d = dict(res.attrs)
            except:
                d = {k: return_h5_str(res, k) for k in columns}
            recall = get_recall(np.array(res["knns"]), gt_knns, 10)
            d['recall'] = recall
            print("{:<8} {:<7} {:<60} => {:1.3f}s {:1.5f}".format(d["data"], d["algo"],  d["params"],d["querytime"], recall))
            writer.writerow(d)

def return_h5_str(f, param):
    if param not in f:
        return 0
    x = f[param][()]
    if type(x) == np.bytes_:
        return x.decode()
    return x

def get_all_results(dirname):
    mask = [dirname + "/*/*.h5"]
    print("search for results matching:")
    print("\n".join(mask))
    for m in mask:
        for fn in glob.iglob(m):
            f = h5py.File(fn, "r")
            if "knns" not in f or not ("size" in f or "size" in f.attrs):
                print("Ignoring " + fn)
                f.close()
                continue
            yield f
            f.close()

def load_h5(path: Path) -> np.ndarray:
    with h5py.File(path, 'r') as f:
        return f['knns'][()]

def get_recall(I, gt, k):
    assert k <= I.shape[1]
    assert len(I) == len(gt)

    n = len(I)
    recall = 0
    for i in range(n):
        recall += len(set(I[i, :k]) & set(gt[i, :k]))
    return recall / (n * k)


if __name__ == '__main__':
    main()
