#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def simulate_bifid_coverage_no_noise(
    num_strains=None,
    total_genes=5500,
    fraction_core=0.3,
    total_reads=100_000,
    random_seed=42,
    multi_copy_fraction=0.05
):
    """
    Simulate read coverage for Bifidobacterium infantis without a coverage noise factor.
    
    Parameters
    ----------
    num_strains : int or None
        Number of strains; if None, randomly choose between 1 and 5.
    total_genes : int
        Total number of genes in the pan-genome (default 5500).
    fraction_core : float
        Fraction of genes that are core (default 0.3).
    total_reads : int
        Total number of simulated reads.
    random_seed : int
        Seed for reproducibility.
    multi_copy_fraction : float
        Fraction of genes that are multi-copy (default 0.05).
        
    Returns
    -------
    coverage_dict : dict
        Dictionary mapping gene IDs to total read coverage.
    strain_proportions : np.array
        The proportion of reads assigned to each strain.
    strain_to_genes : list of lists
        Each element is the list of gene IDs for one strain.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 1. Determine number of strains
    if num_strains is None:
        num_strains = np.random.randint(1, 6)
    
    # 2. Split genes into core and unique
    n_core = int(fraction_core * total_genes)
    n_unique = total_genes - n_core
    core_genes = [f"core_{i}" for i in range(n_core)]
    unique_genes_all = [f"unique_{i}" for i in range(n_unique)]
    
    # 3. Randomly assign each unique gene to one strain
    strain_assignments = np.random.choice(num_strains, size=n_unique)
    unique_genes_per_strain = [[] for _ in range(num_strains)]
    for i, strain_idx in enumerate(strain_assignments):
        unique_genes_per_strain[strain_idx].append(unique_genes_all[i])
    
    # 4. Generate strain proportions from a Dirichlet distribution
    strain_proportions = np.random.dirichlet([1.0] * num_strains)
    
    # 5. Build gene lists for each strain: core genes are present in all strains.
    strain_to_genes = []
    for i in range(num_strains):
        genes_for_strain = list(core_genes) + list(unique_genes_per_strain[i])
        strain_to_genes.append(genes_for_strain)
    
    # 6. For each gene in each strain, assign:
    #    (a) A gene length drawn from a lognormal distribution.
    #    (b) A copy number (default 1, but a fraction may have 2–4 copies).
    strain_to_lengths = []
    strain_to_copy_numbers = []
    
    # Parameters for gene lengths (lognormal distribution)
    length_mean = 7.0    # log-scale mean
    length_sigma = 0.6   # log-scale standard deviation
    
    for i in range(num_strains):
        n_genes_i = len(strain_to_genes[i])
        lengths = np.random.lognormal(mean=length_mean, sigma=length_sigma, size=n_genes_i)
        copy_nums = np.ones(n_genes_i, dtype=int)
        for j in range(n_genes_i):
            if np.random.rand() < multi_copy_fraction:
                # Random copy number from 2 to 4
                copy_nums[j] = np.random.randint(2, 5)
        strain_to_lengths.append(lengths)
        strain_to_copy_numbers.append(copy_nums)
    
    # 7. Allocate reads to strains via a multinomial draw.
    strain_read_counts = np.random.multinomial(total_reads, strain_proportions)
    
    # 8. Within each strain, allocate reads to genes according to their weights.
    #     The weight for gene j in strain i is:
    #         W_{i,j} = (copy number) * (gene length)
    #     The probability for gene j is:
    #         π_{i,j} = W_{i,j} / (Σ_j W_{i,j})
    coverage_arrays = []
    for i in range(num_strains):
        R_i = strain_read_counts[i]
        if R_i == 0:
            coverage_arrays.append(np.zeros(len(strain_to_genes[i]), dtype=int))
            continue
        
        weights = strain_to_copy_numbers[i] * strain_to_lengths[i]
        weights_sum = np.sum(weights)
        if weights_sum == 0:
            coverage_arrays.append(np.zeros(len(strain_to_genes[i]), dtype=int))
            continue
        
        probabilities = weights / weights_sum
        gene_coverages = np.random.multinomial(R_i, probabilities)
        coverage_arrays.append(gene_coverages)
    
    # 9. Build a final coverage dictionary mapping gene ID to its total coverage.
    coverage_dict = {}
    for i in range(num_strains):
        for gene_id, cov in zip(strain_to_genes[i], coverage_arrays[i]):
            # If a core gene appears in every strain, add their coverages
            coverage_dict[gene_id] = coverage_dict.get(gene_id, 0) + cov
    
    return coverage_dict, strain_proportions, strain_to_genes

def main():
    parser = argparse.ArgumentParser(
        description="Simulate read coverage for Bifidobacterium infantis without a coverage noise factor."
    )
    parser.add_argument("--num_strains", type=int, default=None,
                        help="Number of strains (default: random between 1 and 5).")
    parser.add_argument("--total_genes", type=int, default=5500,
                        help="Total number of genes (default: 5500).")
    parser.add_argument("--fraction_core", type=float, default=0.3,
                        help="Fraction of genes that are core (default: 0.3).")
    parser.add_argument("--total_reads", type=int, default=100000,
                        help="Total number of reads to simulate (default: 100000).")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--multi_copy_fraction", type=float, default=0.05,
                        help="Fraction of genes that have multiple copies (default: 0.05).")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to store outputs (default: 'output').")
    args = parser.parse_args()
    
    coverage_dict, proportions, strain_to_genes = simulate_bifid_coverage_no_noise(
        num_strains=args.num_strains,
        total_genes=args.total_genes,
        fraction_core=args.fraction_core,
        total_reads=args.total_reads,
        random_seed=args.random_seed,
        multi_copy_fraction=args.multi_copy_fraction
    )
    
    # Create output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    coverage_values = np.array(list(coverage_dict.values()), dtype=float)
    
    # Save coverage data to CSV
    coverage_csv_path = os.path.join(args.output_dir, "coverage.csv")
    with open(coverage_csv_path, "w") as f:
        f.write("gene_id,coverage\n")
        for gene_id, cov in coverage_dict.items():
            f.write(f"{gene_id},{cov}\n")
    print(f"Coverage data saved to: {coverage_csv_path}")
    
    # Plot histogram with log-spaced bins
    min_cov = max(1, coverage_values.min())
    max_cov = coverage_values.max()
    if max_cov < 1:
        max_cov = 1
    n_bins = 50
    log_bins = np.logspace(np.log10(min_cov), np.log10(max_cov), n_bins)
    
    plt.figure(figsize=(8,6))
    plt.hist(coverage_values, bins=log_bins, edgecolor='black')
    plt.title(f"Simulated Gene Coverage Distribution\n(Strains: {len(proportions)}, Total Reads: {args.total_reads})")
    plt.xlabel("Gene Coverage (reads per gene)")
    plt.ylabel("Number of Genes")
    plt.xscale("log")
    plt.tight_layout()
    
    fig_path = os.path.join(args.output_dir, "coverage_histogram.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Histogram plot saved to: {fig_path}")

if __name__ == "__main__":
    main()