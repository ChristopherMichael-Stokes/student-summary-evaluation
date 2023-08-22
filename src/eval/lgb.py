import pandas as pd

from train.lgb import train_lgb_kfold


def eval_validation(df_train, f_cols, model_params):
    metric_df_content, bst_content = train_lgb_kfold('content', df_train, f_cols, model_params)
    metric_df_wording, bst_wording = train_lgb_kfold('wording', df_train, f_cols, model_params)

    metric_df_content['target'] = 'content'
    metric_df_wording['target'] = 'wording'
    metric_df = pd.concat([metric_df_content, metric_df_wording])
    metric_df = metric_df.loc[['mean', 'std']]
    print(metric_df)

    mcrmse = (metric_df.loc[metric_df.target == 'content', 'rmse'] +
              metric_df.loc[metric_df.target == 'wording', 'rmse']) / 2

    print(f'\nTrain MCRMSE:\t   {mcrmse.iloc[0]}')
    print(f'Validation MCRMSE: {mcrmse.iloc[1]}')
    print(f'Diff:\t {mcrmse.iloc[1]-mcrmse.iloc[0]}\n')

    importance = pd.DataFrame({
        'importance': bst_wording.feature_importance(),
        'feature': bst_wording.feature_name()}).sort_values(by='importance', ascending=False)
    print(importance)
