from pprint import pprint
from typing import Any, Dict, List, Tuple

import pandas as pd

from train.lgb import train_lgb_kfold


def eval_validation(df_train: pd.DataFrame, f_cols: List[str], model_params: Dict[str, Any]) -> Tuple[float, float, float]:
    """Evaluate the average KFold validation performance of an lgb model trained on the 
    supplied feature list with the given parameters

    Args:
        df_train (pd.DataFrame): Dataframe containing all training data
        f_cols (List[str]): List of features to train on
        model_params (Dict[str, Any]): LGB Regressor parameters

    Returns:
        Tuple[float, float, float]: Train set metric, validation set metric & difference between validation and train
    """
    # TODO: create separate parameters for content and wording fcols
    metric_df_content, bst_content = train_lgb_kfold('content', df_train, f_cols, model_params)
    metric_df_wording, bst_wording = train_lgb_kfold('wording', df_train, f_cols, model_params)

    metric_df_content['target'] = 'content'
    metric_df_wording['target'] = 'wording'
    metric_df = pd.concat([metric_df_content, metric_df_wording])
    metric_df = metric_df.loc[['mean', 'std']]
    print(metric_df)

    mcrmse = (metric_df.loc[metric_df.target == 'content', 'rmse'] +
              metric_df.loc[metric_df.target == 'wording', 'rmse']) / 2

    train_mcrmse = float(mcrmse.iloc[0])
    validation_mcrmse = float(mcrmse.iloc[1])
    diff = validation_mcrmse - train_mcrmse
    print(f'\nTrain MCRMSE:\t   {train_mcrmse}')
    print(f'Validation MCRMSE: {validation_mcrmse}')
    print(f'Diff:\t {diff}\n')

    for bst, metric in ((bst_content, 'content'), (bst_wording, 'wording')):

        importance = pd.DataFrame({
            'importance': bst.feature_importance(),
            'feature': bst.feature_name()}).sort_values(by='importance', ascending=False)

        print(f'{metric.capitalize()} Feature importance:')
        pprint(importance)

    return train_mcrmse, validation_mcrmse, diff
