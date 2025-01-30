#!/usr/bin/env python

import argparse
import json
import os
import math
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import pyarrow as pa
import pyarrow.parquet as pq

def parse_args():
    parser = argparse.ArgumentParser(description="Parallel combine coverage files -> Parquet (wide format).")
    parser.add_argument("--config", default="config.json", help="JSON config with 'genes_path'.")
    parser.add_argument("--output", default="gene_coverage.parquet", help="Output Parquet filename.")
    parser.add_argument("--num-cpus", type=int, default=1, help="Number of CPU cores to use.")
    return parser.parse_args()

def read_genes_from_first_file(first_file):
    """Reads gene IDs (in order) from the first coverage file."""
    gene_names = []
    with open(first_file, "r") as f:
        for line in f:
            gene, _ = line.strip().split("\t")
            gene_names.append(gene)
    return gene_names

def read_chunk(args):
    """
    Worker function:
      - Reads each coverage file in 'chunk_files'
      - Builds a local float32 matrix of shape (len(chunk_files), n_genes)
      - Returns (chunk_index, partial_coverage_array)
    """
    chunk_index, chunk_files, n_genes = args
    part_arr = np.zeros((len(chunk_files), n_genes), dtype=np.float32)

    for i, file_path in enumerate(chunk_files):
        with open(file_path, "r") as f:
            line_idx = 0
            for line in f:
                _, cov_str = line.strip().split("\t")
                part_arr[i, line_idx] = float(cov_str)
                line_idx += 1

    return chunk_index, part_arr

def main():
    args = parse_args()

    # 1. Load config and gather coverage files
    with open(args.config, "r") as f:
        config = json.load(f)
    genes_path = config["genes_path"]

    files = [os.path.join(genes_path, f) for f in os.listdir(genes_path)
             if os.path.isfile(os.path.join(genes_path, f))]
    files.sort()
    if not files:
        print("No coverage files found.")
        return

    # 2. Identify gene names from first file
    gene_names = read_genes_from_first_file(files[0])
    n_genes = len(gene_names)
    n_samples = len(files)
    print(f"Found {n_genes} genes and {n_samples} samples.")

    # 3. Prepare to parallel-read in chunks
    num_workers = min(args.num_cpus, cpu_count())
    print(f"Using {num_workers} CPU cores out of {cpu_count()} available.\n")

    chunk_size = math.ceil(n_samples / num_workers)
    chunk_args = []
    for chunk_index in range(num_workers):
        start = chunk_index * chunk_size
        end = min(start + chunk_size, n_samples)
        if start >= n_samples:
            break
        chunk_files = files[start:end]
        chunk_args.append((chunk_index, chunk_files, n_genes))

    # 4. Build the final coverage array
    coverage_data = np.zeros((n_samples, n_genes), dtype=np.float32)

    # 5. Parallel reading
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.map(read_chunk, chunk_args),
                            total=len(chunk_args),
                            desc="Reading Coverage in Parallel"))

    # 6. Place each partial array into coverage_data
    print("Filling coverage_data matrix...")
    for chunk_index, part_arr in sorted(results, key=lambda x: x[0]):
        start = chunk_index * chunk_size
        end = start + part_arr.shape[0]
        coverage_data[start:end, :] = part_arr

    # 7. Build a PyArrow Table (wide: 1 col = sample name, remaining = coverage)
    print("Building Arrow table (wide format)...")
    sample_names = [os.path.basename(fp).split("_")[0] for fp in files]

    arrays = []
    col_names = []

    # First column: Sample (string)
    arrays.append(pa.array(sample_names, type=pa.string()))
    col_names.append("Sample")

    # Next columns: gene coverage
    for j in range(n_genes):
        arrays.append(pa.array(coverage_data[:, j], type=pa.float32()))
        col_names.append(gene_names[j])

    table = pa.Table.from_arrays(arrays, names=col_names)

    # 8. Write to Parquet with compression
    output_path = args.output
    print(f"Writing Parquet -> {output_path}")
    pq.write_table(table, output_path, compression="snappy")  # or compression="gzip"

    print("✅ Done!")

if __name__ == "__main__":
    main()

