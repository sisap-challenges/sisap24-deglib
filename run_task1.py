#!/usr/bin/env python3

import argparse
import gc
import time
from datetime import datetime
from typing import Optional, Callable, Any
from pathlib import Path

import h5py
import numpy as np

import deglib


BUILD_HPARAMS = {
    'edges_per_vertex': 24,
    'extend_k': 64,
    'extend_eps': 0.02,
    'lid': deglib.builder.LID.Low
}


def parse_args():
    parser = argparse.ArgumentParser(description='Run task 1')
    parser.add_argument(
        'dbsize', type=str, choices=["100K", "300K", "10M", "30M", "100M"], help='The database size to use'
    )
    parser.add_argument('-k', type=int, default=30, help='Number of results per query')
    parser.add_argument('--show-progress', action='store_true', help='show progress during graph building')

    return parser.parse_args()


def main():
    # parse DB_SIZE argument
    args = parse_args()
    dbsize: str = args.dbsize
    k: int = args.k

    # create outdir (results-task1/$DBSIZE/$DATE)
    date_str = datetime.now().strftime('yyyymmdd-HHMMSS')
    outdir = Path('results-task1') / dbsize / date_str
    print('outdir: "{}"'.format(outdir))

    # load data (database, queries) using h5-file in batches (convert to f32)
    data_file = Path("data2024") / "laion2B-en-clip768v2-n={}.h5".format(dbsize)
    query_file = Path("data2024") / "public-queries-2024-laion2B-en-clip768v2-n=10k.h5"
    print('data: "{}"\nqueries: "{}"'.format(data_file, query_file))

    # build graph
    callback = 'progress' if args.show_progress else None
    with h5py.File(data_file, 'r') as data_f:
        assert 'emb' in data_f.keys()
        data = data_f['emb']
        build_start_time = time.perf_counter()
        graph = build_deglib_from_data(data, **BUILD_HPARAMS, callback=callback)
        build_end_time = time.perf_counter()
        build_duration = build_end_time - build_start_time

    # benchmark graph
    print('loading queries:')
    queries = load_queries(query_file)
    print('benchmarking graph:')
    benchmark_graph(graph, queries, k, dbsize, build_duration)


def benchmark_graph(graph: deglib.graph.SearchGraph, queries: np.ndarray, k: int, dbsize: str, build_time: float):
    eps_settings = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02]
    for eps in eps_settings:
        start_time = time.perf_counter()
        prediction, distances = graph.search(queries, k=k, eps=eps)
        end_time = time.perf_counter()
        query_time = end_time - start_time
        print('Done searching in {} sec'.format(query_time))

        identifier = f"eps{eps}"
        destination = Path('result') / dbsize / 'deglib_{}.h5'.format(identifier)
        store_results(
            destination, "deglib", 'normal', distances, prediction, build_time, query_time, identifier, dbsize
        )


def build_deglib_from_data(
        data: h5py.Dataset, edges_per_vertex: int = 32, metric: deglib.Metric = deglib.Metric.InnerProduct,
        lid: deglib.builder.LID = deglib.builder.LID.Unknown, extend_k: Optional[int] = None, extend_eps: float = 0.2,
        callback: Callable[[Any], None] | str | None = None
):
    capacity = data.shape[0]
    graph = deglib.graph.SizeBoundedGraph.create_empty(capacity, data.shape[1], edges_per_vertex, metric)
    builder = deglib.builder.EvenRegularGraphBuilder(
        graph, rng=None, lid=lid, extend_k=extend_k, extend_eps=extend_eps, improve_k=0
    )

    labels = np.arange(data.shape[0], dtype=np.uint32)  # TODO: use indices starting at 1
    chunk_size = 100_000
    for min_index in range(0, data.shape[0], chunk_size):
        max_index = min(min_index + chunk_size, data.shape[0])
        builder.add_entry(
            labels[min_index:max_index],
            data[min_index:max_index].astype(np.float32)
        )

    builder.build(callback=callback)

    # remove builder to free memory
    del builder
    gc.collect()

    # TODO: convert to read-only-graph

    return graph


def load_queries(query_file: Path):
    assert query_file.is_file(), 'Could not find query file: {}'.format(query_file)
    with h5py.File(query_file, 'r') as infile:
        assert 'emb' in infile.keys(), 'Could not find "emb" key in query file "{}"'.format(query_file)
        queries = infile['emb'][()]
    return queries


def store_results(dst: Path, algo, kind, distances, result_indices, buildtime, querytime, params, size):
    dst.parent.mkdir(exist_ok=True, parents=True)
    with h5py.File(dst, 'w') as f:
        f.attrs['algo'] = algo
        f.attrs['data'] = kind
        f.attrs['buildtime'] = buildtime
        f.attrs['querytime'] = querytime
        f.attrs['size'] = size
        f.attrs['params'] = params
        f.create_dataset('knns', result_indices.shape, dtype=result_indices.dtype)[:] = result_indices
        f.create_dataset('dists', distances.shape, dtype=distances.dtype)[:] = distances


if __name__ == '__main__':
    main()
