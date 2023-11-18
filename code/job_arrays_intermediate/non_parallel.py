import pandas as pd
from sklearn.model_selection import KFold  # train/test splits
from sklearn.model_selection import GridSearchCV  # selecting
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import plotnine as p9
import time
import statistics


data_path = "/projects/genomic-ml/da2343/ml_project_3/data"

def run_main(n_rows=100):
    # Record the start time
    start_time = time.time()


    # Read the zip file into a pandas dataframe
    zip_df = pd.read_csv(
        f"{data_path}/zip.test.gz",
        header=None,
        sep=" ")
    is01 = zip_df[0].isin([0, 1])
    zip01_df = zip_df.loc[is01, :]
    zip01_shuffled_df = zip01_df.sample(frac=1, random_state=1).reset_index(drop=True)
    
    # Read the spam.csv data into a pandas dataframe
    spam_df = pd.read_csv(
        f"{data_path}/spam.data",
        sep=" ",
        header=None)
    spam_df_shuffled = spam_df.sample(frac=1, random_state=1).reset_index(drop=True)
    
    data_dict = {
        "zip": (zip01_shuffled_df.loc[:, 1:].to_numpy(), zip01_shuffled_df[0]),
        "spam": (spam_df_shuffled.iloc[:, :-1].to_numpy(), spam_df_shuffled.iloc[:, -1])
    }

    test_acc_df_list = []
    for data_set, (input_mat, output_vec) in data_dict.items():
        # take first 100 rows
        input_mat = input_mat[:n_rows, :]
        output_vec = output_vec[:n_rows]
        
        kf = KFold(n_splits=3, shuffle=True, random_state=1)
        for fold_id, indices in enumerate(kf.split(input_mat)):
            index_dict = dict(zip(["train", "test"], indices))
        
            clf = GridSearchCV(estimator=KNeighborsClassifier(),
                            param_grid=[{'n_neighbors': [x]} for x in range(1, 21)], cv=5)
            set_data_dict = {}
            for set_name, index_vec in index_dict.items():
                set_data_dict[set_name] = {
                    "X": input_mat[index_vec],
                    "y": output_vec.iloc[index_vec]
                }
            # ** is unpacking a dict to use as the named arguments
            # clf.fit(X=set_data_dict["train"]["X"], y=set_data_dict["train"]["y"]])
            clf.fit(**set_data_dict["train"])

            pipe = make_pipeline(
                StandardScaler(), LogisticRegressionCV(cv=5, max_iter=1_000) )
            pipe.fit(**set_data_dict["train"])
            
            featureless = DummyClassifier(strategy="most_frequent")
            featureless.fit(**set_data_dict["train"])

            pred_dict = {
                "nearest_neighbors": clf.predict(set_data_dict["test"]["X"]),
                "linear_model": pipe.predict(set_data_dict["test"]["X"]),
                "featureless": featureless.predict(set_data_dict["test"]["X"])
            }
            for algorithm, pred_vec in pred_dict.items():
                test_acc_dict = {
                    "test_accuracy_percent": (
                        pred_vec == set_data_dict["test"]["y"]).mean()*100,
                    "data_set": data_set,
                    "fold_id": fold_id,
                    "algorithm": algorithm
                }
                test_acc_df_list.append(pd.DataFrame(test_acc_dict, index=[0]))

    test_acc_df = pd.concat(test_acc_df_list)
    # gg = p9.ggplot() +\
    #     p9.geom_point(
    #         p9.aes(
    #             x="test_accuracy_percent",
    #             y="algorithm"            
    #         ),
    #         data=test_acc_df) +\
    #     p9.facet_wrap("data_set")
    # gg.save("non_parallel_algo_acc.png")

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the result
    # print(f"Elapsed time: {elapsed_time} seconds")
    return elapsed_time


n_row_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
n_row_time_list = []
for i in n_row_list:
    time_list = []
    for j in range(3):
        elapsed_time = run_main(i)
        time_list.append(elapsed_time)
        
    mean_time = statistics.mean(time_list)
    # print(f"Mean time: {mean_time} seconds")
    std_time = statistics.stdev(time_list)
    # print(f"Standard deviation of time: {std_time} seconds")
    n_row_time_list.append({
        "n_rows": i,
        "mean_time": mean_time,
        "std_time": std_time,
        "type": "non_parallel"
    })

n_row_time_df = pd.DataFrame(n_row_time_list)
print(n_row_time_df)
# save to csv
n_row_time_df.to_csv("non_parallel_n_row_time_df.csv", index=False)