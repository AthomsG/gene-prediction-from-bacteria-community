import json
import os
import re
import pandas as pd
from tqdm import tqdm

# Load paths from JSON file
def load_paths(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
    return data

# Function to extract species and coverage from a file
def extract_species_coverage(file_path):
    species_pattern = re.compile(r's__([\w\s-]+)')  # Capture species names correctly
    sample_data = {}

    with open(file_path, "r") as file:
        next(file)  # Skip header
        for line in file:
            columns = line.strip().split("\t")  # Tab-separated
            if len(columns) > 2:
                coverage = float(columns[1])  # Convert coverage to float
                match = species_pattern.search(columns[2])  # Find species name
                if match:
                    species_name = match.group(1).strip()  # Extract species name
                    sample_data[species_name] = sample_data.get(species_name, 0) + coverage  # Sum coverage

    return sample_data

# Main function to process all files
if __name__ == "__main__":
    # Load config.json
    config_file = "config.json"
    paths = load_paths(config_file)
    species_path = paths.get("species_path")

    if not species_path:
        print("Error: 'species_path' not found in config.json.")
        exit(1)

    # List all files in the species path
    files = [f for f in os.listdir(species_path) if os.path.isfile(os.path.join(species_path, f))]

    if not files:
        print("No files found in the species directory.")
        exit(1)

    # Dictionary to store sample data
    data = {}

    for file in tqdm(files, desc="Processing Samples"):
        file_path = os.path.join(species_path, file)
        species_coverage = extract_species_coverage(file_path)
        data[file[:-4]] = species_coverage

    df = pd.DataFrame.from_dict(data, orient="index").fillna(0)

    output_file = "species_coverage.parquet"
    df.to_parquet(output_file)

    print(f"✅Species coverage saved as: {output_file}")
