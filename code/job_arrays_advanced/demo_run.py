import sys
import os
import pandas as pd
from sklearn.model_selection import KFold  # train/test splits
from sklearn.model_selection import GridSearchCV  # selecting
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import plotnine as p9
import time
import statistics


params_df = pd.read_csv("params.csv")

if len(sys.argv) == 2:
    prog_name, task_str = sys.argv
    param_row = int(task_str)
else:
    print("len(sys.argv)=%d so trying first param" % len(sys.argv))
    param_row = 0

data_path = "/projects/genomic-ml/da2343/ml_project_3/data"
param_dict = dict(params_df.iloc[param_row, :])

dataset_name = param_dict["dataset_name"]
algorithm = param_dict["algorithm"]

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

algo_dict = {
    "KNeighborsClassifier": GridSearchCV(estimator=KNeighborsClassifier(),
                            param_grid=[{'n_neighbors': [x]} for x in range(1, 21)], cv=5),
    "LinearModel": make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, max_iter=1000)),
    "Featureless": DummyClassifier(strategy="most_frequent"),
}
classifier = algo_dict[algorithm]
data_set = data_dict[dataset_name]

input_mat, output_vec = data_set
test_acc_df_list = []
kf = KFold(n_splits=3, shuffle=True, random_state=1)
for fold_id, indices in enumerate(kf.split(input_mat)):
    index_dict = dict(zip(["train", "test"], indices))
    set_data_dict = {}
    for set_name, index_vec in index_dict.items():
        set_data_dict[set_name] = {
            "X": input_mat[index_vec],
            "y": output_vec.iloc[index_vec]
        }
    classifier.fit(**set_data_dict["train"])
    pred_vec = classifier.predict(set_data_dict["test"]["X"])
    accuracy = accuracy_score(set_data_dict["test"]["y"], pred_vec)
    test_acc_df_list.append(pd.DataFrame({
            "test_accuracy_percent": accuracy * 100,
            "data_set": dataset_name,
            "fold_id": fold_id,
            "algorithm": algorithm
        }, index=[0]))

test_acc_df = pd.concat(test_acc_df_list)
# print(test_acc_df)
# Save dataframe as a csv to output directory
out_file = f"results/{param_row}.csv"
test_acc_df.to_csv(out_file, encoding='utf-8', index=False)
print("Done!!")

