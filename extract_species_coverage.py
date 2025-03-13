import json
import os
import pandas as pd
from tqdm import tqdm

# Load paths from JSON file
def load_paths(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
    return data

# Function to extract coverage for a given taxonomic rank from a file,
# while excluding lines that contain any taxa from the exclusions list,
# even if that taxon appears deeper than the taxon_rank.
def extract_taxon_coverage(file_path, taxon_rank="s__", exclusions=None):
    """
    Reads a tab-separated file and sums the coverage for each taxon at the specified rank,
    skipping any lines where ANY token in the taxonomy (regardless of its rank) matches an exclusion.

    Args:
        file_path (str): Path to the input file.
        taxon_rank (str): Taxonomic rank prefix to aggregate on (e.g., "s__" for species,
                          "g__" for genus, "p__" for phylum, etc.).
        exclusions (list): List of taxon names (without any prefix) to ignore.
                           This can refer to a deeper level than taxon_rank.

    Returns:
        dict: Dictionary with taxon names (at the specified rank) as keys and summed coverage as values.
    """
    if exclusions is None:
        exclusions = []
    sample_data = {}

    with open(file_path, "r") as file:
        next(file)  # Skip header
        for line in file:
            columns = line.strip().split("\t")
            if len(columns) > 2:
                try:
                    coverage = float(columns[1])
                except ValueError:
                    continue  # Skip lines with invalid coverage

                taxonomy_str = columns[2]
                # Split the taxonomy string by semicolon and remove extra spaces
                tokens = [token.strip() for token in taxonomy_str.split(";")]

                # Check every token in the taxonomy.
                # If any token (after removing its rank prefix) is in the exclusions list, skip this line.
                skip_line = False
                for token in tokens:
                    if "__" in token:
                        # Split on the first occurrence of "__"
                        parts = token.split("__", 1)
                        if len(parts) > 1:
                            name = parts[1].strip()
                            if name in exclusions:
                                skip_line = True
                                break
                if skip_line:
                    continue

                # Now, find the token corresponding to the desired taxon_rank
                for token in tokens:
                    if token.startswith(taxon_rank):
                        # Remove the rank prefix to get the taxon name
                        taxon_name = token[len(taxon_rank):].strip()
                        sample_data[taxon_name] = sample_data.get(taxon_name, 0) + coverage
                        break  # Only use the first occurrence for each line
    return sample_data

if __name__ == "__main__":
    # Specify which taxonomic rank you want to aggregate.
    # For example, "g__" for genus.
    taxon_rank = "g__" 

    # List of taxon names to exclude.
    # Even if you are aggregating at genus level, if any token in the taxonomy (e.g., s__)
    # matches an exclusion (like "Bifidobacterium infantis"), the entire line is skipped.
    exclusions = ["Bifidobacterium infantis"]

    # Load config.json to get the path to your taxonomy files
    config_file = "config.json"
    paths = load_paths(config_file)
    species_path = paths.get("species_path")

    if not species_path:
        print("Error: 'species_path' not found in config.json.")
        exit(1)

    # List all files in the species directory
    files = [f for f in os.listdir(species_path) if os.path.isfile(os.path.join(species_path, f))]
    if not files:
        print("No files found in the species directory.")
        exit(1)

    # Dictionary to store aggregated data from all samples
    data = {}
    for file in tqdm(files, desc="Processing Samples"):
        file_path = os.path.join(species_path, file)
        taxon_coverage = extract_taxon_coverage(file_path, taxon_rank=taxon_rank, exclusions=exclusions)
        # Remove file extension for naming
        data[file[:-4]] = taxon_coverage

    # Create a DataFrame from the dictionary and fill missing values with 0
    df = pd.DataFrame.from_dict(data, orient="index").fillna(0)

    # Save the results to a Parquet file; the filename reflects the taxon rank used
    output_file = f"taxon_coverage_{taxon_rank.replace('__', '')}.parquet"
    df.to_parquet(output_file)

    print(f"✅ Taxon coverage saved as: {output_file}")