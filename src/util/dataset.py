from pandas import DataFrame
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (GridSearchCV, ShuffleSplit,
                                     cross_val_predict, cross_val_score,
                                     cross_validate)


def load_dataset():
    df: DataFrame = load_breast_cancer(as_frame=True)
    return df.data, df.target

# def preprocess_data(data: DataFrame):
    
