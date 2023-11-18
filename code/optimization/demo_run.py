import pandas as pd
import numpy as np
import torch
import plotnine as p9
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier



class TorchModel(torch.nn.Module):
    def __init__(self, n_hidden_layers, units_in_first_layer, units_per_hidden_layer=100):
        super(TorchModel, self).__init__()
        units_per_layer = [units_in_first_layer]
        for layer_i in range(n_hidden_layers):
            units_per_layer.append(units_per_hidden_layer)
        units_per_layer.append(1)
        seq_args = []
        for layer_i in range(len(units_per_layer)-1):
            units_in = units_per_layer[layer_i]
            units_out = units_per_layer[layer_i+1]
            seq_args.append(
                torch.nn.Linear(units_in, units_out))
            if layer_i != len(units_per_layer)-2:
                seq_args.append(torch.nn.ReLU())
        self.stack = torch.nn.Sequential(*seq_args)

    def forward(self, feature_mat):
        return self.stack(feature_mat)


class NumpyData(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, item):
        return self.features[item, :], self.labels[item]

    def __len__(self):
        return len(self.labels)


class MyCV:
    def __init__(self, estimator, param_grid, cv):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv

    def fit(self, X, y):
        self.train_features = X
        self.train_labels = y
        self.best_params_ = {}
        np.random.seed(1)
        fold_vec = np.random.randint(
            low=0, high=self.cv, size=self.train_labels.size)
        
        best_mean_accuracy = 0
        loss_df_list = []
        for param_dict in self.param_grid:
            for param_name, [param_value] in param_dict.items():
                setattr(self.estimator, param_name, param_value)
            
            local_loss_df_list = []    
            for test_fold in range(self.cv):
                is_set_dict = {
                    "validation": fold_vec == test_fold,
                    "subtrain": fold_vec != test_fold,
                }
                set_features = {
                    set_name: self.train_features[is_set, :]
                    for set_name, is_set in is_set_dict.items()
                }
                set_labels = {
                    set_name: self.train_labels[is_set]
                    for set_name, is_set in is_set_dict.items()
                }
                self.estimator.fit(
                    X=set_features, y=set_labels)
                predicted_labels = self.estimator.predict(
                    X=set_features["validation"])
           
                local_loss_df_list.append(self.estimator.loss_df)
            mean_local_loss_df = pd.concat(local_loss_df_list).groupby(
                ["hidden_layers", "step_size", "epoch", "set_name"]).mean().reset_index()
            loss_df_list.append(mean_local_loss_df)         
        self.loss_mean_df = pd.concat(loss_df_list)
        print(self.loss_mean_df)

    def predict(self, X):
        return self.estimator.predict(X)


class RegularizedMLP:
    def __init__(self, max_epochs, batch_size, step_size, hidden_layers, units_per_hidden_layer):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.hidden_layers = hidden_layers
        self.units_per_hidden_layer = units_per_hidden_layer
        self.loss_fun = torch.nn.BCEWithLogitsLoss()

    def fit(self, X, y):
        set_features = X
        set_labels = y
        # Preparing subtrain and validation data loaders
        subtrain_csv = NumpyData(
            set_features["subtrain"], set_labels["subtrain"])
        subtrain_dl = torch.utils.data.DataLoader(
            subtrain_csv, batch_size=self.batch_size, shuffle=True)
        loss_df_list = []
       
        model = TorchModel(self.hidden_layers, set_features["subtrain"].shape[1])
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.step_size)

        for epoch in range(self.max_epochs):
            for batch_features, batch_labels in subtrain_dl:
                # Take a step and compute prediction error
                # Compute prediction error
                pred_tensor = model(batch_features.float()).reshape(
                    len(batch_labels.float()))
                loss_tensor = self.loss_fun(
                    pred_tensor, batch_labels.float())
                # Backpropagation
                optimizer.zero_grad()
                loss_tensor.backward()
                optimizer.step()

            # then compute subtrain/validation loss.
            for set_name in set_features:
                feature_mat = set_features[set_name]
                label_vec = set_labels[set_name]
                feature_mat_tensor = torch.from_numpy(
                    feature_mat.astype(np.float32))
                label_vec_tensor = torch.from_numpy(
                    label_vec.astype(np.float32))

                pred_tensor = model(feature_mat_tensor.float()).reshape(
                    len(label_vec_tensor.float()))
                loss_tensor = self.loss_fun(
                    pred_tensor, label_vec_tensor.float())
                set_loss = loss_tensor.item()

                loss_df_list.append(pd.DataFrame({
                    "hidden_layers": self.hidden_layers,
                    "step_size": self.step_size,
                    "set_name": set_name,
                    "loss": set_loss,
                    "epoch": epoch,
                }, index=[0]))
        self.model = model
        self.loss_df = pd.concat(loss_df_list)

    def decision_function(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.Tensor(X)).numpy().ravel()

    def predict(self, X):
        return np.where(self.decision_function(X) > 0, 1, 0)

data_path = "/projects/genomic-ml/da2343/ml_project_3/data"

spam_df = pd.read_csv(
    f"{data_path}/spam.data",
    header=None,
    sep=" ")

spam_features = spam_df.iloc[:, :-1].to_numpy()
spam_scaled_features = (
    spam_features - spam_features.mean(axis=0)) / spam_features.std(axis=0)
spam_labels = spam_df.iloc[:, -1].to_numpy()


data_dict = {
    "spam": (spam_scaled_features, spam_labels),
    # "zip": (zip_features, zip_labels),
}


def hyperparameter_training_and_diagnostic_plot():
    for data_set, (input_mat, output_vec) in data_dict.items():
        # param_dicts = [{'hidden_layers': [L]} for L in range(1, 5)]
        
        param_dicts = [
            {
                'hidden_layers': [L],
                'step_size': [U],
            }
            for L in range(1, 3)
            for U in [0.1, 0.2]
        ]
        print(param_dicts)
            
        rmlp = RegularizedMLP(
            max_epochs=100,
            batch_size=100,
            step_size=0.1,
            hidden_layers=3,
            units_per_hidden_layer=100,
        )
        learner_instance = MyCV(estimator=rmlp, param_grid=param_dicts, cv=2)
        learner_instance.fit(input_mat, output_vec)
        loss_df = learner_instance.loss_mean_df
        
        # print(loss_df)
        
        loss_df.index = range(len(loss_df))
        
        set_colors = {"subtrain": "red", "validation": "blue"}
        validation_df = loss_df.query("set_name=='validation'")
        min_i = validation_df.loss.argmin()
        min_row = pd.DataFrame(dict(validation_df.iloc[min_i, :]), index=[0])
        gg = p9.ggplot() +\
            p9.facet_grid(". ~ hidden_layers+step_size") +\
            p9.scale_color_manual(values=set_colors) +\
            p9.scale_fill_manual(values=set_colors) +\
            p9.geom_line(
                p9.aes(
                    x="epoch",
                    y="loss",
                    color="set_name"
                ),
                data=loss_df) +\
            p9.geom_point(
                p9.aes(
                    x="epoch",
                    y="loss",
                    fill="set_name"
                ),
                color="black",
                data=min_row)
        gg.save(f"{data_set}_01.png", width=10, height=5)


hyperparameter_training_and_diagnostic_plot()