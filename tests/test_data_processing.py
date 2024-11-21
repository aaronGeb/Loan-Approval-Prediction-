import unittest
import pandas as pd
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))
from data_processing import DataProcessing


class TestDataProcessing(unittest.TestCase):
    """
            A class for unit-testing function in the data_processing.py file

            Args:
    -----
                    unittest.TestCase this allows the new class to inherit
                    from the unittest module


    def setUp(self):
        self.data = pd.read_csv(
            "/Users/Aaron/Loan-Approval-Prediction-/data/processed/credit_risk_dataset.csv"
        )
        self.data_processing = DataProcessing(self.data)

    def test_read_data(self):
        self.assertEqual(
            self.data_processing.read_data(
                "../data/processed/credit_risk_dataset.csv"
            ).shape,
            (32581, 12),
        )

    def test_remove_duplicates(self):
        self.data_processing.remove_duplicates()
        self.assertEqual(self.data.duplicated().sum(), 0)

    def test_standardize_column_names(self):
        self.data_processing.standardize_column_names(self.data)
        expected_columns = [
            "person_age",
            "person_income",
            "person_home_ownership",
            "person_emp_length",
            "loan_intent",
            "loan_grade",
            "loan_amnt",
            "loan_int_rate",
            "loan_status",
            "loan_percent_income",
            "cb_person_default_on_file",
            "cb_person_cred_hist_length",
        ]
        self.assertEqual(self.data.columns.tolist(), expected_columns)
    """

    def setUp(self):
        self.data = pd.DataFrame(
            {
                "age": [25, np.nan, 35],
                "income": [50000, 60000, np.nan],
                "loan_amount": [20000, 30000, 40000],
            }
        )
        self.processor = DataProcessing(self.data)

    def test_fill_null_values_skewness(self):
        self.processor.fill_null_values_skewness(self.data, "income")
        self.assertFalse(self.data["income"].isnull().any())


if __name__ == "__main__":
    unittest.main()
