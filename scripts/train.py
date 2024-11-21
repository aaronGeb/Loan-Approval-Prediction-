#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import Optional
import pickle
import os

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class LoanModel:
    def __init__(
        self, data: DataFrame, n_folds: int, output_file="../models/loan_model.pkl"
    ):
        self.data = data
        self.model = None
        self.dv = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.output_file = output_file
        self.n_folds = n_folds

    def load_data(self, path: str) -> DataFrame:
        """
        Load data from a csv file
        Args:
          file_path: str: path to the csv file
        Returns:
          DataFrame: data loaded from the csv file
        """
        self.data = pd.read_csv(path)
        return self.data

    def split_data(self):
        """
        Split the data into training and testing set
        """
        X = self.data.drop("loan_status", axis=1)
        y = self.data["loan_status"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def train_model(self):
        """
        Train the model
        """
        self.dv = DictVectorizer(sparse=False)
        self.X_train = self.dv.fit_transform(self.X_train.to_dict(orient="records"))
        self.model = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=10,
            min_samples_leaf=1,
            min_samples_split=15,
            class_weight="balanced",
        )

        self.model.fit(self.X_train, self.y_train)

    def cross_validate(self):
        """
        Perform cross-validation
        """
        kf = KFold(n_splits=self.n_folds)
        scores = []
        for train_index, test_index in kf.split(self.data):
            train = self.data.iloc[train_index]
            test = self.data.iloc[test_index]
            X_train = train.drop("loan_status", axis=1)
            y_train = train["loan_status"]
            X_test = test.drop("loan_status", axis=1)
            y_test = test["loan_status"]
            dv = DictVectorizer(sparse=False)
            X_train = dv.fit_transform(X_train.to_dict(orient="records"))
            model = DecisionTreeClassifier(
                criterion="entropy",
                max_depth=10,
                min_samples_leaf=1,
                min_samples_split=15,
            )
            model.fit(X_train, y_train)
            X_test = dv.transform(X_test.to_dict(orient="records"))
            y_pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))
        print(f"Cross-validation scores: {scores}")
        print(f"Mean accuracy: {np.mean(scores)}")

    def evaluate_model(self):
        """
        Evaluate the model
        """
        if isinstance(self.X_test, np.ndarray):
            self.X_test = pd.DataFrame(self.X_test)
        self.X_test = self.dv.transform(self.X_test.to_dict(orient="records"))
        y_pred = self.model.predict(self.X_test)
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred)}")
        print(f"Classification report: {classification_report(self.y_test, y_pred)}")
        print(f"Confusion matrix: {confusion_matrix(self.y_test, y_pred)}")

    def save_model(self):
        """
        Save the model
        """

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, "wb") as file:
            pickle.dump((self.model, self.dv), file)
        print("Saving model to", self.output_file)

    def load_model(self):
        """
        Load the model adn DictVectorizer
        """
        with open(self.output_file, "rb") as file:
            self.model, self.dv = pickle.load(file)
        print("Model loaded successfully")
        print(f"feature names after loading : {self.dv.feature_names_}")
        return self.model, self.dv

    def debug_data_shapes(self):
        """
        Print debug information about data shapes and types
        """
        print(f"X_train shape: {self.X_train.shape}")
        print(f"y_train shape: {self.y_train.shape}")
        print(f"X_test shape: {self.X_test.shape}")
        print(f"y_test shape: {self.y_test.shape}")
        print(
            f"Feature names: {self.dv.feature_names_ if self.dv else 'DictVectorizer not initialized'}"
        )


if __name__ == "__main__":
    model = LoanModel(data=None, n_folds=5)
    model.load_data(
        "/Users/Aaron/Loan-Approval-Prediction-/data/processed/credit_risk_dataset.csv"
    )
    model.split_data()
    model.debug_data_shapes()
    model.save_model()
    model.train_model()
    model.cross_validate()
    model.evaluate_model()
    model.save_model()
    model.load_model()
    model.evaluate_model()
