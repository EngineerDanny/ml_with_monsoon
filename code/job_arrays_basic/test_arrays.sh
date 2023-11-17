#!/bin/bash
#SBATCH --array=0-9
#SBATCH --time=1:00:00
#SBATCH --mem=1GB
#SBATCH --cpus-per-task=1
#SBATCH --error=/projects/genomic-ml/da2343/ml_project_3/code/slurm-%A_%a.out
#SBATCH --output=/projects/genomic-ml/da2343/ml_project_3/code/slurm-%A_%a.out
#SBATCH --job-name=test_arrays
cd /projects/genomic-ml/da2343/ml_project_3/code
python test_arrays.py $SLURM_ARRAY_TASK_ID