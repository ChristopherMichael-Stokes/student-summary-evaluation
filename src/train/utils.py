from math import sqrt

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_errors(y, y_pred):
    return {
        'r2': r2_score(y, y_pred),
        'rmse': sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred)
    }


def select_columns(df: pd.DataFrame):
    numeric_features = df.select_dtypes(include=np.number)
    target_columns = ['content', 'wording']
    feature_columns = [col for col in numeric_features if col not in target_columns]

    targets = numeric_features[target_columns]
    features = numeric_features[feature_columns]
    prompt_group = pd.Categorical(df['prompt_title'])
    return targets, features, prompt_group
