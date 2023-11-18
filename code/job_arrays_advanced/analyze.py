# combines the results for each parameter combination into a single result for analysis/plotting.
import pandas as pd
from glob import glob
import plotnine as p9

date_time = "2023-11-17_20:57"
out_df_list = []
for out_csv in glob(f"/scratch/da2343/ml_project_3_{date_time}/results/*.csv"):
    out_df_list.append(pd.read_csv(out_csv))
test_acc_df = pd.concat(out_df_list)

root_results_dir = "/projects/genomic-ml/da2343/ml_project_3/code/job_arrays_advanced/results"
test_acc_df.to_csv(f"{root_results_dir}/{date_time}_results.csv", index=False)


# create the plot
gg = p9.ggplot() +\
    p9.geom_point(
        p9.aes(
            x="test_accuracy_percent",
            y="algorithm"            
        ),
        data=test_acc_df) +\
    p9.facet_wrap("data_set")
gg.save("parallel_algo_acc.png", width=5, height=5, dpi=1000)


