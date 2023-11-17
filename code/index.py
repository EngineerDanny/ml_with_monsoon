import pandas as pd
from sklearn.model_selection import KFold  # train/test splits
from sklearn.model_selection import GridSearchCV  # selecting
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
import plotnine as p9

# Read the zip file into a pandas dataframe
zip_df = pd.read_csv(
    "./data/zip.test.gz",
    header=None,
    sep=" ")

print(zip_df.shape)

is01 = zip_df[0].isin([0, 1])
zip01_df = zip_df.loc[is01, :]

# Read the spam.csv data into a pandas dataframe
spam_df = pd.read_csv(
    "./data/spam.data",
    sep=" ",
    header=None)


data_dict = {
    "zip": (zip01_df.loc[:, 1:].to_numpy(), zip01_df[0]),
    "spam": (spam_df.iloc[:, :-1].to_numpy(), spam_df.iloc[:, -1])
}

test_acc_df_list = []
for data_set, (input_mat, output_vec) in data_dict.items():
    kf = KFold(n_splits=3, shuffle=True, random_state=1)
    for fold_id, indices in enumerate(kf.split(input_mat)):
        print("fold_id = " + str(fold_id))
        index_dict = dict(zip(["train", "test"], indices))
        param_dicts = [{'n_neighbors': [x]} for x in range(1, 21)]
        # does subtrain/validation splits.

        clf = GridSearchCV(estimator=KNeighborsClassifier(),
                           param_grid=param_dicts, cv=5)
    
        # method 2: dict instead of tuple.
        set_data_dict = {}
        for set_name, index_vec in index_dict.items():
            set_data_dict[set_name] = {
                "X": input_mat[index_vec],
                "y": output_vec.iloc[index_vec]
            }
        # ** is unpacking a dict to use as the named arguments
        # clf.fit(X=set_data_dict["train"]["X"], y=set_data_dict["train"]["y"]])
        clf.fit(**set_data_dict["train"])

        # print out the best parameters
        print("best params = " + str(clf.best_params_))

        pipe = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000))
        pipe.fit(**set_data_dict["train"])

        cv_df = pd.DataFrame(clf.cv_results_)
        cv_df.loc[:, ["param_n_neighbors", "mean_test_score"]]

        print(cv_df.loc[:, ["param_n_neighbors", "mean_test_score"]])

        # for each n_neighbors, make a plot that shows the mean_test_score as a function of a
        # hyper-parameter for one or more algorithms and/or data sets.
        # plot a curve of the mean_test_score as a function of n_neighbors
        plot = p9.ggplot(
            data=cv_df,
            mapping=p9.aes(
                x="param_n_neighbors",
                y="mean_test_score")
        ) + p9.geom_point() + p9.ggtitle(f"{data_set}_fold_{fold_id}_cv")

        plot.save(
            filename=f"./{data_set}_fold_{fold_id}_cv.png",
            width=6,
            height=6,
            dpi=300
        )


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

# make a ggplot to visually examine which learning algorithm is best for each data set. Use
# geom_point with x=”test_accuracy_percent”, y=”algorithm”, and facet_grid(“. ~ data_set”).

# make a facetted plot with one panel per image
gg = p9.ggplot() +\
    p9.geom_point(
        p9.aes(
            x="test_accuracy_percent",
            y="algorithm"
        ),
        data=test_acc_df) +\
    p9.facet_wrap("data_set")
gg.save("./facetted.png")