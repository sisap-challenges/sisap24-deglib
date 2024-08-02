#!/usr/bin/env python3

import os

# set tensorflow cpu threads, before importing tensorflow
os.environ["OMP_NUM_THREADS"] = "4"

import multiprocessing
import argparse
import gc
import time
from typing import Optional, Callable, Any
from pathlib import Path

import h5py
import numpy as np

from compress import CompressionNet

import deglib


BUILD_HPARAMS = {
    'edges_per_vertex': 32,
    'lid': deglib.builder.LID.Low,
    'extend_k': 64,
    'extend_eps': 0.1,
}

EPS_SETTINGS = [
    0.0,  0.001, 0.002, 0.005, 0.01,
    0.02, 0.03,  0.04,  0.05,  0.06,
    0.07, 0.08,  0.09,  0.1,   0.11,
    0.12, 0.13,  0.14,  0.15,  0.16,
    0.17, 0.18,  0.19,  0.2,   0.21,
    0.22, 0.25,  0.27,  0.3,   0.35,
]


def parse_args():
    parser = argparse.ArgumentParser(description='Run task 1')
    parser.add_argument(
        'dbsize', type=str, choices=["100K", "300K", "10M", "30M", "100M"], help='The database size to use'
    )
    parser.add_argument('-k', type=int, default=30, help='Number of results per query')
    parser.add_argument('--show-progress', action='store_true', help='show progress during graph building')
    parser.add_argument(
        '--compression', '-c', type=int, default=0, help='use compression net to reduce data dimensionality'
    )

    return parser.parse_args()


def main():
    # parse DB_SIZE argument
    args = parse_args()
    dbsize: str = args.dbsize
    k: int = args.k

    # load data (database, queries) using h5-file in batches (convert to f32)
    data_file = Path("data2024") / "laion2B-en-clip768v2-n={}.h5".format(dbsize)
    query_file = Path("data2024") / "public-queries-2024-laion2B-en-clip768v2-n=10k.h5"
    print('data: "{}"\nqueries: "{}"'.format(data_file, query_file))
    
    # load compression network
    comp_net = None
    if args.compression:
        comp_net = CompressionNet(target_dim=args.compression)

    # build graph
    callback = 'progress' if args.show_progress else None
    with h5py.File(data_file, 'r') as data_f:
        assert 'emb' in data_f.keys()
        data = data_f['emb']
        build_start_time = time.perf_counter()
        graph = build_deglib_from_data(data, comp_net, **BUILD_HPARAMS, callback=callback)
        build_end_time = time.perf_counter()
        build_duration = build_end_time - build_start_time

    # benchmark graph
    print('loading queries:')
    queries = load_queries(query_file)
    
    print('benchmarking graph:')
    benchmark_graph(graph, queries, comp_net, k, dbsize, build_duration)


def benchmark_graph(
        graph: deglib.graph.SearchGraph, queries: np.ndarray, comp_net: Optional[CompressionNet], k: int, dbsize: str,
        build_time: float
):
    print('queries:', queries.shape, queries.dtype)
    print(f'{"eps":<8} {"query time":<13} {"comp time":<13} {"graph time":<13}')
    for eps in EPS_SETTINGS:
        start_time = time.perf_counter()
        if comp_net is not None:
            compressed_queries = comp_net.compress(
                queries, quantize=True, batch_size=queries.shape[0]
            )
        else:
            compressed_queries = queries
        start_time_benchmark = time.perf_counter()
        prediction, distances = graph.search(
            compressed_queries, k=k, eps=eps, threads=multiprocessing.cpu_count(), thread_batch_size=32
        )
        end_time = time.perf_counter()
        query_time = end_time - start_time
        print(f'{eps:<8} {query_time:<13.4f} {start_time_benchmark - start_time:<13.4f} '
              f'{end_time - start_time_benchmark:<13.4f}')

        identifier = f"eps{eps}"
        destination = Path('result') / dbsize / 'deglib_{}.h5'.format(identifier)
        # offset prediction to start at index 1
        prediction += 1
        store_results(
            destination, "deglib", 'normal', distances, prediction, build_time, query_time, identifier, dbsize
        )


def build_deglib_from_data(
        data: h5py.Dataset, comp_net: Optional[CompressionNet], edges_per_vertex: int,
        lid: deglib.builder.LID, extend_k: Optional[int] = None, extend_eps: float = 0.2,
        callback: Callable[[Any], None] | str | None = None
):
    print('building graph with: {}'.format(BUILD_HPARAMS))
    quantize = True
    num_samples = data.shape[0]
    if comp_net is not None:
        dim = comp_net.target_dim
        metric = deglib.Metric.L2_Uint8 if quantize else deglib.Metric.L2
    else:
        dim = data.shape[1]
        metric = deglib.Metric.InnerProduct
    print('creating empty graph', flush=True)
    graph = deglib.graph.SizeBoundedGraph.create_empty(num_samples, dim, edges_per_vertex, metric)
    print('creating builder', flush=True)
    builder = deglib.builder.EvenRegularGraphBuilder(
        graph, rng=None, lid=lid, extend_k=extend_k, extend_eps=extend_eps, improve_k=0
    )

    labels = np.arange(num_samples, dtype=np.uint32)
    chunk_size = 50_000
    print('adding data to builder', flush=True)
    for counter, min_index in enumerate(range(0, num_samples, chunk_size)):
        max_index = min(min_index + chunk_size, num_samples)
        chunk = data[min_index:max_index].astype(np.float32)
        if counter % 10 == 0:
            print('adding features {}'.format(min_index), flush=True)
        if comp_net is not None:
            chunk = comp_net.compress(chunk, quantize=quantize, batch_size=chunk.shape[0])
        else:
            chunk = chunk.astype(np.float32)
        builder.add_entry(
            labels[min_index:max_index],
            chunk
        )

    print('builder.build():', flush=True)
    builder.build(callback=callback)

    # remove builder to free memory
    del builder
    gc.collect()

    graph.remove_non_mrng_edges()
    graph = deglib.graph.ReadOnlyGraph.from_graph(graph)
    gc.collect()

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
