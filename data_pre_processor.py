import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def count_cardinality(data_frame):
    column_cardinality = {}
    for column in data_frame.columns:
        cardinality = data_frame[column].nunique()
        column_cardinality[column] = cardinality
    return column_cardinality


def count_empty_values(data_frame):
    return data_frame.isnull().sum()


def count_duplicated_values(data_frame):
    return data_frame.duplicated().sum()


def remove_rows_with_missing_values(data_frame):
    # Drop rows with any missing values
    cleaned_data_frame = data_frame.dropna()

    return cleaned_data_frame


def split_data(df, test_size=0.2, random_state=42):
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_data, test_data


def data_normalizer(train_data, test_data):
    scaler = MinMaxScaler()
    numerical_columns = train_data.select_dtypes(include=['number']).columns

    # Scale all numerical columns and replace the values in the DataFrame
    train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
    test_data[numerical_columns] = scaler.fit_transform(test_data[numerical_columns])

    return train_data, test_data

