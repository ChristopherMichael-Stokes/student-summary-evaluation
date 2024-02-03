from pathlib import Path

import hydra
import mlflow

from eval import eval_validation
from features import build_features

# import xformers


@hydra.main('../conf', 'conf.yaml', version_base='1.2')
def main(cfg):
    # Read and join data paths from config
    project_root = Path(hydra.utils.get_original_cwd()).absolute()
    data_dir = project_root / cfg.data.dir
    sample_submission = data_dir / cfg.data.sample_submission
    summaries_train = data_dir / cfg.data.summaries_train
    summaries_test = data_dir / cfg.data.summaries_test
    prompts_train = data_dir / cfg.data.prompts_train
    prompts_test = data_dir / cfg.data.prompts_test

    # Create defined split
    if cfg.run.load_split == 'train':
        df = build_features.make_split(summaries_path=summaries_train,
                                       prompts_path=prompts_train, make_pos_features=cfg.run.make_pos_features)
    elif cfg.run.load_split == 'test':
        df = build_features.make_split(summaries_path=summaries_test,
                                       prompts_path=prompts_test, make_pos_features=cfg.run.make_pos_features)
    else:
        raise AssertionError('Invalid dataset split')

    mlflow.set_tracking_uri(project_root / cfg.mlflow.uri)
    with mlflow.start_run():
        for param, value in cfg.train.model_params.items():
            mlflow.log_param(param, value)

        train_mcrmse, validation_mcrmse, diff = eval_validation(
            df_train=df, f_cols=cfg.train.f_cols, model_params=dict(cfg.train.model_params))

        mlflow.log_metric('Train MCRMSE', train_mcrmse)
        mlflow.log_metric('Validation MCRMSE', validation_mcrmse)
        mlflow.log_metric('Train / Val diff', diff)


if __name__ == '__main__':
    main()
