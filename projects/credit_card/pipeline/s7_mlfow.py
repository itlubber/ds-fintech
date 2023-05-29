from pathlib import Path
import time
import datetime
import mlflow

import config
from projects.credit_card import model_config
from util import bm_helper, data_helper, lgbm_helper, report_helper


def run_lgbm():
    df_woe = data_helper.Data.load("df_woe", prefix=model_config.prefix)
    woe = data_helper.Data.load("woe", prefix=model_config.prefix)

    features = sorted(set(woe.selected_features) - {model_config.pk, model_config.label, "sample_type"})
    df_train = df_woe.loc[df_woe["sample_type"] == "train"]
    df_valid = df_woe.loc[df_woe["sample_type"] == "test"]
    df_xtrain, df_ytrain = df_train[features], df_train[model_config.label]
    df_xvalid, df_yvalid = df_valid[features], df_valid[model_config.label]

    mlflow.set_tracking_uri(f"{Path(config.ROOT_DIR, 'data', 'mlruns')}")

    experiment_name = "credi_card"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except Exception as e:
        print(e)
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    tag = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y%m%d-%H:%M:%S')
    run_name = f'run_{tag}'
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        mlflow.set_tag("purpose", "toy_sample")

        model = lgbm_helper.LightGBM()
        model.select_and_fit(
            df_xtrain,
            df_ytrain,
            df_xvalid=df_xvalid,
            df_yvalid=df_yvalid,
            n_estimators=500,
        )
        data_helper.Data.dump("lgbm_model", model, prefix=model_config.prefix)
        df_importance = model.get_importance()

        df_woe["lgbm_score"] = model.predict(df_woe)
        data_helper.Data.dump("df_lgbm_result", df_woe, prefix=model_config.prefix)

        df_report = report_helper.ModelReport.get_report(df_woe, "sample_type", "lgbm_score", model_config.label)


        params = model.model.get_params()
        for k, v in params.items():
            mlflow.log_param(k, v)

        for sample_type, metrics in df_report.to_dict().items():
            for metric, value in metrics.items():
                mlflow.log_metric(f"{sample_type}_{metric}", value)

        fp_report = str(Path(config.ROOT_DIR, 'data', model_config.prefix, 'report.csv'))
        df_report.to_csv(fp_report)

        fp_importance = str(Path(config.ROOT_DIR, 'data', model_config.prefix, 'feature_importance.csv'))
        df_importance.to_csv(fp_importance)

        mlflow.log_artifact(str(Path(config.ROOT_DIR, 'data', model_config.prefix, 'lgbm_model.pkl')))
        mlflow.log_artifact(str(Path(config.ROOT_DIR, 'data', model_config.prefix, 'df_lgbm_result.pkl')))
        mlflow.log_artifact(str(Path(config.ROOT_DIR, 'data', model_config.prefix, 'woe.pkl')))
        mlflow.log_artifact(fp_report)
        mlflow.log_artifact(fp_importance)


def main():
    run_lgbm()


if __name__ == "__main__":
    main()
