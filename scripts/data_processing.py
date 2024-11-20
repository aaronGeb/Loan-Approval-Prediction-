#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import Optional


class DataProcessing:
    def __init__(self, data: Optional[DataFrame] = None):
        self.data = data

    def read_data(self, path: str) -> DataFrame:
        """
        Load data from a csv file
        Args:
            file_path: str: path to the csv file
        Returns:
            DataFrame: data loaded from the csv file
        """
        self.data = pd.read_csv(path)
        return self.data

    def fill_null_values_skewness(self, data: DataFrame, columns: str) -> DataFrame:
        """
        calculate the skewness of the data and fill the null values with the
        median if the skewness is greater than 1
        Args:
            data: DataFrame: data with null values to be filled
           columns: str: columns to be filled
        Returns:
            DataFrame: data with null values filled
        """
        skewness_value = self.data[columns].skew()
        if skewness_value > 1:
            self.data[columns].fillna(self.data[columns].median(), inplace=True)
        else:
            self.data[columns].fillna(self.data[columns].mean(), inplace=True)
        return self.data

    def remove_duplicates(self) -> DataFrame:
        """
        Remove duplicates from the data
        Returns:
            DataFrame: data with duplicates removed
        """
        self.data = self.data.drop_duplicates()
        return self.data

    def standardize_column_names(self, data: DataFrame) -> DataFrame:
        """
        Standardize the columns name of the data
        Args:
            data: DataFrame: data with columns to be standardized
        Returns:
            DataFrame: standardized data
        """
        self.data = self.data.rename(
            columns=lambda x: x.strip()
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        return self.data

    def rename_observation_column(self, data: DataFrame) -> DataFrame:
        """
        Rename the observation column
        Args:
            data: DataFrame: data with observation column to be renamed
        Returns:
            DataFrame: data with observation column renamed
        """
        for col in self.data.columns:
            if self.data[col].dtype == "object":
                self.data[col] = (
                    self.data.data[col]
                    .str.strip()
                    .str.lower()
                    .str.replace(" ", "_")
                    .str.replace("(", "")
                    .str.replace(")", "")
                )

        return self.data

    def transform_numerical_features(self, data) -> DataFrame:
        """
        Apply transformation to numerical features based on the skewness
         of the data
        - Log transformation for highly skewed data
        - Square root transformation for moderate skewed features
        - Standardized for slightly skewed or symmetric feature.

        Args:
            data: DataFrame: data with numerical features to be transformed
        Returns:
            DataFrame: data with numerical features transformed
        """
        numerical_features = self.data.select_dtypes(indlude=np.number).columns
        for feature in numerical_features:
            skewness = self.data[feature].skew()
            if skewness > 1:
                self.data[feature] = np.log1p(self.data[feature])
            elif 0.5 < skewness <= 1:
                self.data[feature] = np.sqrt(self.data[feature])
            else:
                self.data[feature] = (
                    self.data[feature] - self.data[feature].mean()
                ) / self.data[feature].std()
        return self.data
