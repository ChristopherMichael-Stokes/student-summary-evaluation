from typing import List, Tuple

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GroupKFold

from .utils import calculate_errors, select_columns


def train_lgb_kfold(
        target: str,
        df_train: pd.DataFrame,
        feature_names: List[str],
        model_params: dict) -> Tuple[pd.DataFrame, lgb.Booster]:

    targets, features, prompt_group = select_columns(df_train)

    group_kfold = GroupKFold(n_splits=prompt_group.unique().size)
    assert group_kfold.get_n_splits(features, targets, prompt_group) == len(prompt_group.unique())

    train_errors, val_errors = [], []
    for i, (train_index, test_index) in enumerate(group_kfold.split(features, targets, prompt_group)):
        X_train = features[feature_names].iloc[train_index]
        y_train = targets.iloc[train_index][target]

        X_val = features[feature_names].iloc[test_index]
        y_val = targets.iloc[test_index][target]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, y_val)
        bst = lgb.train(model_params,
                        train_set=train_data, valid_sets=[train_data, val_data],
                        valid_names=['fit', 'val'], callbacks=[lgb.log_evaluation(100)])

        train_errors.append(calculate_errors(y_train, bst.predict(X_train)))
        val_errors.append(calculate_errors(y_val, bst.predict(X_val)))

    train_metrics = pd.DataFrame.from_records(train_errors).describe()
    train_metrics['set'] = 'train'
    val_metrics = pd.DataFrame.from_records(val_errors).describe()
    val_metrics['set'] = 'val'
    metric_df = pd.concat([train_metrics, val_metrics])

    return metric_df, bst
