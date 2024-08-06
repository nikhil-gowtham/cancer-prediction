# import libraries
from typing import Any

import mlflow
from numpy import ndarray
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class TrainModel:
    """
    Class for model training operations
    """

    def __init__(self, data_path: str) -> None:
        """
        Load dataset
        Split data to train and test

        Args:
            data_path (str): path to dataset
        """
        self.data = self.load_data(data_path=data_path)
        self.X_train, self.X_test, self.y_train, self.y_test = (
            self.train_test_split()
        )

    def load_data(self, data_path) -> None:
        """
        Load data from csv to dataframe
        """
        return pd.read_csv(data_path)

    def train_test_split(self) -> tuple:
        """
        Split data to train and test
        80% to train and 20% to test
        """
        # get features
        X = self.data.drop(["diagnosis"], axis=1)

        # get label
        y = self.data["diagnosis"]

        # split to train and test
        return train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    def calculate_log_loss(
        self, y_train_pred: ndarray, y_test_pred: ndarray
    ) -> tuple:
        """
        Calculate train and test log loss

        Args:
            y_train_pred (ndarray): predicted probablities of train data
            y_test_pred (ndarray): predicted probablities of test data

        Returns:
            tuple: train log loss and test log loss
        """
        print("printing type")
        print(type(y_train_pred))
        train_log_loss = log_loss(self.y_train, y_train_pred)
        test_log_loss = log_loss(self.y_test, y_test_pred)

        return train_log_loss, test_log_loss

    def train_model(
        self, model_name: str, model: Any, param_grid: dict
    ) -> None:
        """
        Use Grid search and train model
        Log data to MLFlow

        Args:
            model_name (str): name of model
            model (Any): model object
            param_grid (dict): parameters for grid search
        """
        try:
            # start MLflow run
            with mlflow.start_run():
                mlflow.set_tracking_uri("http://127.0.0.1:5000")

                # set up the GridSearchCV
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=5,
                    scoring="balanced_accuracy",
                )

                # fit GridSearchCV to the training data
                grid_search.fit(self.X_train, self.y_train)

                # log the best parameters
                best_params = grid_search.best_params_
                mlflow.log_params(best_params)

                # log the best model
                best_model = grid_search.best_estimator_
                mlflow.sklearn.log_model(
                    best_model,
                    f"model_{model_name}",
                    registered_model_name=f"model_{model_name}",
                )

                # log metrics, we will use log_loss
                y_train_pred = best_model.predict_proba(self.X_train)
                y_test_pred = best_model.predict_proba(self.X_test)

                train_log_loss, test_log_loss = self.calculate_log_loss(
                    y_train_pred=y_train_pred, y_test_pred=y_test_pred
                )

                mlflow.log_metric("train_log_loss", train_log_loss)
                mlflow.log_metric("test_log_loss", test_log_loss)

                print(f"-----{model_name} metrics-----")
                print("train_log_loss: ", train_log_loss)
                print("test_log_loss: ", test_log_loss)
                print("----------")

        except Exception as ex:
            print("Exception during model training: ", ex)

    def train_knn(self):
        """
        Train k-nearest neighbours
        """
        # create a kNN classifier
        knn = KNeighborsClassifier()

        # define the parameter grid to search
        param_grid = {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        }

        # train and log best model
        self.train_model(model_name="knn", model=knn, param_grid=param_grid)

    def train_logistic_regression(self):
        """
        Train logistic regression
        """
        # create a Logistic Regression classifier
        log_reg = LogisticRegression(max_iter=1000)

        # define the parameter grid to search
        param_grid = {
            "penalty": ["l1", "l2", "none"],
            "C": [0.01, 0.1, 1, 10],
            "solver": ["newton-cg", "lbfgs", "liblinear", "saga"],
        }

        # train and log best model
        self.train_model(
            model_name="logistic_regression",
            model=log_reg,
            param_grid=param_grid,
        )

    def train_random_forest(self):
        """
        Train random forest
        """
        # create a Random Forest classifier
        rf = RandomForestClassifier(n_jobs=-1)

        # define the parameter grid to search
        param_grid = {
            "n_estimators": [10, 50, 100, 200],
            "max_features": ["auto", "sqrt", "log2"],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
        }

        # train and log best model
        self.train_model(
            model_name="random_forest", model=rf, param_grid=param_grid
        )

    def train_svm(self):
        """
        Train SVM
        """
        # create an SVM classifier
        svm = SVC(probability=True)

        # define the parameter grid to search
        param_grid = {
            "C": [0.01, 0.1, 1, 10, 100],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "degree": [2, 3, 4, 5],
            "gamma": ["scale", "auto"],
        }

        # train and log best model
        self.train_model(model_name="svm", model=svm, param_grid=param_grid)


if __name__ == "__main__":
    # initialize TrainModel object
    model_training = TrainModel(
        r"C:\Users\nikhi\Personal\bits\3\mlops\assignment1\cancer-prediction\data\Cancer_Data.csv"
    )

    # knn
    print("Training k-NN")
    model_training.train_knn()
    print("Training k-NN completed")

    # logistic regression
    print("Training logistic regression")
    model_training.train_logistic_regression()
    print("Training logistic regression completed")

    # random forest
    print("Training random forest")
    model_training.train_random_forest()
    print("Training random forest  completed")

    # SVM
    print("Training svm")
    model_training.train_svm()
    print("Training svm completed")
