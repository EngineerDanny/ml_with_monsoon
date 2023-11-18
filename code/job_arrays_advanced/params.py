
from datetime import datetime
import pandas as pd
import numpy as np
import os
import shutil
import sys
import pandas as pd


root_data_dir = "/projects/genomic-ml/da2343/ml_project_2/data"
params_df_list = []

algo_list = ["KNeighborsClassifier", 
            "LinearModel", 
            "Featureless"]
dataset_list = ["zip","spam"]

for dataset_name in dataset_list:
    params_dict = {
        'dataset_name': [dataset_name],
        'algorithm': algo_list,
    }

    params_df = pd.MultiIndex.from_product(
        params_dict.values(),
        names=params_dict.keys()
    ).to_frame().reset_index(drop=True)
    params_df_list.append(params_df)
params_concat_df = pd.concat(params_df_list, ignore_index=True)
n_tasks, ncol = params_concat_df.shape


date_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
job_name = f"ml_project_3_{date_time}"
job_dir = "/scratch/da2343/" + job_name
results_dir = os.path.join(job_dir, "results")
os.system("mkdir -p " + results_dir)
params_concat_df.to_csv(os.path.join(job_dir, "params.csv"), index=False)

run_one_contents = f"""#!/bin/bash
#SBATCH --array=0-{n_tasks-1}
#SBATCH --time=1:00:00
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --error={job_dir}/slurm-%A_%a.out
#SBATCH --output={job_dir}/slurm-%A_%a.out
#SBATCH --job-name={job_name}
cd {job_dir}
python run_one.py $SLURM_ARRAY_TASK_ID
"""
run_one_sh = os.path.join(job_dir, "run_one.sh")
with open(run_one_sh, "w") as run_one_f:
    run_one_f.write(run_one_contents)

run_orig_py = "demo_run.py"
run_one_py = os.path.join(job_dir, "run_one.py")
shutil.copyfile(run_orig_py, run_one_py)
orig_dir = os.path.dirname(run_orig_py)
orig_results = os.path.join(orig_dir, "results")
os.system("mkdir -p " + orig_results)
orig_csv = os.path.join(orig_dir, "params.csv")
params_concat_df.to_csv(orig_csv, index=False)

msg = f"""created params CSV files and job scripts, test with
python {run_orig_py}
SLURM_ARRAY_TASK_ID=0 bash {run_one_sh}"""
print(msg)


