#!/usr/bin/env python

import argparse
import json
import os
import math
import numpy as np
import re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import pyarrow as pa
import pyarrow.parquet as pq

def parse_args():
    parser = argparse.ArgumentParser(description="Parallel combine species coverage files -> Parquet.")
    parser.add_argument("--config", default="config.json", help="JSON config with 'species_path'.")
    parser.add_argument("--output", default="species_coverage.parquet", help="Output Parquet filename.")
    parser.add_argument("--num-cpus", type=int, default=1, help="Number of CPU cores to use.")
    return parser.parse_args()

def extract_species_from_first_file(first_file):
    """Extract unique species names (columns) from the first file."""
    species_pattern = re.compile(r's__([\w\s-]+)')  # Regex to capture species names
    species_names = set()

    with open(first_file, "r") as f:
        next(f)  # Skip header
        for line in f:
            columns = line.strip().split("\t")
            if len(columns) > 2:
                match = species_pattern.search(columns[2])
                if match:
                    species_names.add(match.group(1).strip())  # Extract species name

    return sorted(species_names)  # Keep consistent order across all samples

def process_chunk(args):
    """
    Worker function:
    - Reads each species coverage file in `chunk_files`
    - Builds a NumPy matrix of shape (len(chunk_files), n_species)
    - Returns (chunk_index, partial_coverage_array)
    """
    chunk_index, chunk_files, species_list = args
    n_species = len(species_list)
    species_idx_map = {species: i for i, species in enumerate(species_list)}

    part_arr = np.zeros((len(chunk_files), n_species), dtype=np.float32)
    species_pattern = re.compile(r's__([\w\s-]+)')

    for i, file_path in enumerate(chunk_files):
        row_idx = i
        with open(file_path, "r") as f:
            next(f)  # Skip header
            for line in f:
                columns = line.strip().split("\t")
                if len(columns) > 2:
                    coverage = float(columns[1])
                    match = species_pattern.search(columns[2])
                    if match:
                        species_name = match.group(1).strip()
                        if species_name in species_idx_map:
                            part_arr[row_idx, species_idx_map[species_name]] += coverage

    return chunk_index, part_arr

def main():
    args = parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)
    species_path = config["species_path"]

    # Collect and sort species coverage files
    files = [os.path.join(species_path, f) for f in os.listdir(species_path)
             if os.path.isfile(os.path.join(species_path, f))]
    files.sort()
    if not files:
        print("No species coverage files found.")
        return

    # Extract species names from the first file
    species_list = extract_species_from_first_file(files[0])
    n_species = len(species_list)
    n_samples = len(files)
    print(f"Found {n_species} species and {n_samples} samples.")

    # Prepare for parallel processing
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
        chunk_args.append((chunk_index, chunk_files, species_list))

    # Build the final coverage matrix
    coverage_data = np.zeros((n_samples, n_species), dtype=np.float32)

    # Parallel processing
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.map(process_chunk, chunk_args),
                            total=len(chunk_args),
                            desc="Processing Species Coverage in Parallel"))

    # Insert each chunk into the final coverage matrix
    print("Filling coverage_data matrix...")
    for chunk_index, part_arr in sorted(results, key=lambda x: x[0]):
        start = chunk_index * chunk_size
        end = start + part_arr.shape[0]
        coverage_data[start:end, :] = part_arr

    # Convert to PyArrow Table and write to Parquet
    print("Building Arrow table (wide format)...")
    sample_names = [os.path.basename(fp).split("_")[0][:-4] for fp in files]

    arrays = []
    col_names = []

    # First column: Sample (string)
    arrays.append(pa.array(sample_names, type=pa.string()))
    col_names.append("Sample")

    # Next columns: species coverage
    for j in range(n_species):
        arrays.append(pa.array(coverage_data[:, j], type=pa.float32()))
        col_names.append(species_list[j])

    table = pa.Table.from_arrays(arrays, names=col_names)

    # Write to Parquet with compression
    output_path = args.output
    print(f"Writing Parquet -> {output_path}")
    pq.write_table(table, output_path, compression="snappy")  # Use "gzip" for better compression

    print("✅ Done!")

if __name__ == "__main__":
    main()

