import pandas as pd
from sklearn.model_selection import KFold  # train/test splits
from sklearn.model_selection import GridSearchCV  # selecting
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import plotnine as p9


data_path = "/projects/genomic-ml/da2343/ml_project_3/data"

# Read the zip file into a pandas dataframe
zip_df = pd.read_csv(
    f"{data_path}/zip.test.gz",
    header=None,
    sep=" ")

is01 = zip_df[0].isin([0, 1])
zip01_df = zip_df.loc[is01, :]
# Read the spam.csv data into a pandas dataframe
spam_df = pd.read_csv(
    f"{data_path}/spam.data",
    sep=" ",
    header=None)


data_dict = {
    "zip": (zip01_df.loc[:, 1:].to_numpy(), zip01_df[0]),
    "spam": (spam_df.iloc[:, :-1].to_numpy(), spam_df.iloc[:, -1])
}

algo_dict = {
    "nearest_neighbors": KNeighborsClassifier(),
    "linear_model": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)),
    "featureless": DummyClassifier(strategy="most_frequent"),
}
    

test_acc_df_list = []
for data_set, (input_mat, output_vec) in data_dict.items():
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
            StandardScaler(), LogisticRegression(max_iter=1000))
        pipe.fit(**set_data_dict["train"])

        pred_dict = {
            "nearest_neighbors": clf.predict(set_data_dict["test"]["X"]),
            "linear_model": pipe.predict(set_data_dict["test"]["X"]),
            "featureless": set_data_dict["train"]["y"].value_counts().idxmax()
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
print(test_acc_df)

# make a ggplot to visually examine which learning algorithm is best for each data set.
gg = p9.ggplot() +\
    p9.geom_point(
        p9.aes(
            x="test_accuracy_percent",
            y="algorithm"            
        ),
        data=test_acc_df) +\
    p9.facet_wrap("data_set")
gg.save("non_parallel_algo_acc.png")