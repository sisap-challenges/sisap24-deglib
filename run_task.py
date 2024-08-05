#!/usr/bin/env python3

import multiprocessing
import argparse
import gc
import time
from typing import Optional, Callable, Any, List
from pathlib import Path

import h5py
import numpy as np

from compress import CompressionNet

import deglib


BUILD_HPARAMS = {
    'quantize': True,
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
    parser = argparse.ArgumentParser(description='Run task 1 with --compression=512 and task 3 with --compression=64')
    parser.add_argument(
        '--dbsize',
        type=str,
        choices=["100K", "300K", "10M", "30M", "100M"],
        help='The database size to use'
    )
    parser.add_argument(
        '-k',
        type=int,
        default=30,
        help='Number of results per query'
    )
    parser.add_argument(
        '--show-progress',
        default=True,
        action='store_true',
        help='show progress during graph building'
    )
    parser.add_argument(
        '--compression', '-c',
        type=int,
        default=0,
        help='use compression net to reduce data dimensionality'
    )
    parser.add_argument(
        '--query-file',
        type=Path,
        default=Path("data") / "public-queries-2024-laion2B-en-clip768v2-n=10k.h5",
        help='The query file'
    )

    return parser.parse_args()


def main():
    # parse DB_SIZE argument
    args = parse_args()
    dbsize: str = args.dbsize
    k: int = args.k

    # load data (database, queries) using h5-file in batches (convert to f32)
    data_file = Path("data") / "laion2B-en-clip768v2-n={}.h5".format(dbsize)
    query_file = args.query_file
    print('Use the following files:\ndata: "{}"\nqueries: "{}"\n'.format(data_file, query_file))
    
    # load compression network
    comp_net = None
    if args.compression:
        print('Load compression network {}D'.format(args.compression))
        comp_net = CompressionNet(target_dim=args.compression)

    # build graph
    callback = 'progress' if args.show_progress else None
    with h5py.File(data_file, 'r') as data_f:
        assert 'emb' in data_f.keys()
        data = data_f['emb']
        build_start_time = time.perf_counter()
        graph = build_deglib_from_data(data, comp_net, **BUILD_HPARAMS, callback=callback)
        build_duration = time.perf_counter() - build_start_time

    # benchmark graph
    print('\nStart benchmarking the graph:')

    # load cluster centers
    cluster_centers = np.load('cluster_centers.npy', allow_pickle=True)
    if comp_net is not None:
        cluster_centers = comp_net.compress(cluster_centers, quantize=BUILD_HPARAMS['quantize'], batch_size=cluster_centers.shape[0])
    entry_indices, _ = graph.search(
        cluster_centers, eps=0.2, k=1, threads=min(multiprocessing.cpu_count(), cluster_centers.shape[0]),
        thread_batch_size=1
    )
    entry_indices = list(entry_indices.flatten())
    print('Seed vertex indices for the evaluation: {}'.format(entry_indices))

    # evaluate on a test set
    queries = load_queries(query_file)
    benchmark_graph(graph, queries, comp_net, k, dbsize, build_duration, entry_indices)


def build_deglib_from_data(
        data: h5py.Dataset, comp_net: Optional[CompressionNet], quantize: bool, edges_per_vertex: int,
        lid: deglib.builder.LID, extend_k: Optional[int] = None, extend_eps: float = 0.2,
        callback: Callable[[Any], None] | str | None = None
):
    print('\n\nBuilding graph with hyperparameters: {}'.format(BUILD_HPARAMS))
    num_samples = data.shape[0]
    if comp_net is not None:
        dim = comp_net.target_dim
        metric = deglib.Metric.L2_Uint8 if quantize else deglib.Metric.L2
    else:
        dim = data.shape[1]
        metric = deglib.Metric.InnerProduct
    graph = deglib.graph.SizeBoundedGraph.create_empty(num_samples, dim, edges_per_vertex, metric)
    builder = deglib.builder.EvenRegularGraphBuilder(
        graph, rng=None, lid=lid, extend_k=extend_k, extend_eps=extend_eps, improve_k=0
    )

    print(f"Start adding {num_samples} data points to builder", flush=True)
    chunk_size = 50_000
    start_time = time.perf_counter()
    labels = np.arange(num_samples, dtype=np.uint32)
    for counter, min_index in enumerate(range(0, num_samples, chunk_size)):
        if counter != 0 and counter % 10 == 0:
            print('Added {} data points after {:5.1f}s'.format(min_index, time.perf_counter() - start_time), flush=True)

        max_index = min(min_index + chunk_size, num_samples)
        chunk = data[min_index:max_index]
        if comp_net is not None:
            chunk = comp_net.compress(chunk, quantize=quantize, batch_size=chunk.shape[0])
        else:
            chunk = chunk.astype(np.float32)
        builder.add_entry(
            labels[min_index:max_index],
            chunk
        )
    print('Added {} data points after {:5.1f}s\n'.format(num_samples, time.perf_counter() - start_time), flush=True)

    print('Start building graph:', flush=True)
    builder.build(callback=callback)

    # remove builder to free memory
    del builder
    gc.collect()

    print('Removing all none MRNG conform edges ... ', flush=True)
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


def benchmark_graph(
        graph: deglib.graph.SearchGraph, queries: np.ndarray, comp_net: Optional[CompressionNet], k: int, dbsize: str,
        build_time: float, entry_indices: List[int] | None
):
    print('queries:', queries.shape, queries.dtype)
    print(f'{"eps":<8} {"query time":<13} {"comp time":<13} {"graph time":<13}')
    for eps in EPS_SETTINGS:
        start_time_benchmark = time.perf_counter()
        if comp_net is not None:
            compressed_queries = comp_net.compress(
                queries, quantize=BUILD_HPARAMS['quantize'], batch_size=queries.shape[0]
            )
        else:
            compressed_queries = queries
        start_time_search = time.perf_counter()
        prediction, distances = graph.search(
            compressed_queries, k=k, eps=eps, threads=multiprocessing.cpu_count(), thread_batch_size=32,
            entry_vertex_indices=entry_indices
        )
        end_time = time.perf_counter()
        query_time = end_time - start_time_benchmark
        print(f'{eps:<8} {query_time:<13.4f} {start_time_search - start_time_benchmark:<13.4f} '
              f'{end_time - start_time_search:<13.4f}')

        # store the information
        # https://github.com/sisap-challenges/sisap23-laion-challenge-evaluation/tree/0a6f90debe73365abee210d3950efc07223c846d
        algo = "deglib"
        data = "normal"
        if comp_net is not None:
            data = "uint{}".format(comp_net.target_dim)
        index_identifier = "epv={}_extendK={}_extendEps={}".format(BUILD_HPARAMS['edges_per_vertex'], BUILD_HPARAMS['extend_k'], BUILD_HPARAMS['extend_eps'])
        identifier = f"index=({index_identifier}),query=(eps={eps})"
        destination = Path('results') / dbsize / data / 'deglib_{}.h5'.format(identifier)
        prediction += 1     # offset prediction to start at index 1
        store_results(
            destination, algo, data, distances, prediction, build_time, query_time, identifier, dbsize
        )


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
