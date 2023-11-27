# Machine Learning with Monsoon (NAUs Supercomputer)
This repository shows the code to explain some of the basic concepts you will need in your machine learning workflow 
using monsoon. 

## Connecting to Monsoon
- Access through the monsoon [dashboard](https://ondemand.hpc.nau.edu/pun/sys/dashboard/)
- Access through the secure shell (ssh). On your terminal, run below and type your password:
```bash
ssh -Y <username>@monsoon.hpc.nau.edu
```
## Interactive/Debug Work
- Request a compute node with 4GB of RAM and 1 cpu for 24 hours
```bash
srun -t 24:00:00 --mem=4GB --cpus-per-task=1 --pty bash
```
- Request a compute node with 8GB of RAM and 1 cpu for 1 hour
```bash
srun -t 1:00:00 --mem=8GB --cpus-per-task=1 python analysis.py
```

## Submitting Jobs
You can also write your program and submit a job shell script for you to be placed in the queue.
An example job script (jobscript.sh) is:
```bash
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/scratch/da2343/output.txt #SBATCH --error=/scratch/da2343/error.txt #SBATCH --time=20:00
#SBATCH --mem=1GB
#SBATCH --cpus-per-task=1
python analysis.py
```
Submit the job script using:
```bash
sbatch jobscript.sh
```


## [Time Graph](https://github.com/EngineerDanny/ml_with_monsoon/tree/main/code/job_arrays_intermediate)
<img src="https://github.com/EngineerDanny/ml_with_monsoon/blob/main/code/job_arrays_intermediate/time_graph.png" 
  alt="time_graph" 
  title="time_graph"
  width="700px"
  height="500px">

## [Algorithm Selection](https://github.com/EngineerDanny/ml_with_monsoon/tree/main/code/job_arrays_advanced)
<img src="https://github.com/EngineerDanny/ml_with_monsoon/blob/main/code/job_arrays_advanced/parallel_algo_acc.png" 
  alt="parallel_algo_acc" 
  title="parallel_algo_acc"
  width="700px"
  height="500px">

## [Hyper-Parameter Tuning](https://github.com/EngineerDanny/ml_with_monsoon/tree/main/code/optimization)
<img src="https://github.com/EngineerDanny/ml_with_monsoon/blob/main/code/optimization/loss_df_01.png" 
  alt="loss_df_01" 
  title="loss_df_01">
