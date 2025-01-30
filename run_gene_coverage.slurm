#!/bin/bash
#SBATCH --job-name=gene_coverage
#SBATCH --output=logs/gene_coverage_%j.out
#SBATCH --error=logs/gene_coverage_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=thomas.gaehtgens@univie.ac.at

cd /lisc/user/gaehtgens/github/gene-prediction-from-bacteria-community
mkdir -p logs

source venv/bin/activate

echo "Node: $(hostname)"
echo "Python: $(python --version)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# Pass the same CPU count to the Python script
python extract_genes_coverage.py \
  --config config.json \
  --output gene_coverage.parquet \
  --num-cpus "$SLURM_CPUS_PER_TASK" \
  2>&1 | tee logs/gene_coverage_"$SLURM_JOB_ID".out

echo "✅ Job completed!"

