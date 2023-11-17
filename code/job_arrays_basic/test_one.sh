#!/bin/bash
#SBATCH --array=0-4
#SBATCH --time=1:00:00
#SBATCH --mem=8MB
#SBATCH --cpus-per-task=1
#SBATCH --error=/projects/genomic-ml/da2343/ml_project_3/code/job_arrays_basic/slurm-%A_%a.out
#SBATCH --output=/projects/genomic-ml/da2343/ml_project_3/code/job_arrays_basic/slurm-%A_%a.out
#SBATCH --job-name=job_arrays_basic
cd /projects/genomic-ml/da2343/ml_project_3/code/job_arrays_basic
python test_one.py $SLURM_ARRAY_TASK_ID






