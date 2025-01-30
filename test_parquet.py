import pyarrow.parquet as pq

# File paths
gene_parquet = "processed_data/gene_coverage.parquet"
species_parquet = "processed_data/species_coverage.parquet"

# Function to read and print first 5 rows of a Parquet file
def print_parquet_head(file_path, label):
    print(f"\n🔹 Loading {label} Parquet file: {file_path}")
    table = pq.read_table(file_path)
    print(table.slice(0, 5).to_pandas())  # Efficiently print first 5 rows

# Read and print both files
print_parquet_head(gene_parquet, "Gene Coverage")
print_parquet_head(species_parquet, "Species Coverage")

