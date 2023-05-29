import ast
import csv
from pathlib import Path
from timeit import default_timer as timer

import lightgbm as lgb
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import StratifiedKFold

import config
from projects.credit_card import model_config
from util import data_helper, metric_helper

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)


class BO:
    def __init__(self, fp_path, **kwargs):
        self.fp_path = fp_path
        self.iter = 0
        self.train_set = None

        self.kfold = kwargs.get("kfold", 3)

        csv_conn = open(self.fp_path, "w")
        writer = csv.writer(csv_conn)
        writer.writerow(["loss", "auc_train", "auc_valid", "params", "iteration", "train_time"])
        csv_conn.close()

    def load_data(self, df_xtrain, df_ytrain):
        self.df_xtrain = df_xtrain.reset_index(drop=True)
        self.df_ytrain = df_ytrain.reset_index(drop=True)

    def objective(self, params):
        self.iter += 1

        start = timer()
        model = lgb.LGBMClassifier(
            **params,
            n_estimators=200,
            objective="cross_entropy",
            class_weight="balanced",
            importance_type="gain",
            boosting_type="gbdt",
            silent=True,
            n_jobs=1,
            random_state=19910908
        )

        lst_auc_train, lst_auc_valid = list(), list()
        kf = StratifiedKFold(n_splits=self.kfold, shuffle=False)
        for itrain, ivalid in kf.split(self.df_xtrain, self.df_ytrain):
            df_xtrain, df_ytrain = (
                self.df_xtrain.loc[itrain, :],
                self.df_ytrain.loc[itrain],
            )
            df_xvalid, df_yvalid = (
                self.df_xtrain.loc[ivalid, :],
                self.df_ytrain.loc[ivalid],
            )

            eval_set = [(df_xtrain, df_ytrain), (df_xvalid, df_yvalid)]
            model.fit(df_xtrain, df_ytrain, eval_set=eval_set, eval_metric="auc", verbose=0)

            auc_train = metric_helper.Metric.get_auc(df_ytrain, model.predict(df_xtrain))
            auc_valid = metric_helper.Metric.get_auc(df_yvalid, model.predict(df_xvalid))
            lst_auc_train.append(auc_train)
            lst_auc_valid.append(auc_valid)

        run_time = timer() - start

        auc_train_avg = np.mean(lst_auc_train)
        auc_valid_avg = np.mean(lst_auc_valid)
        loss = -np.mean(lst_auc_valid)

        csv_conn = open(self.fp_path, "a")
        writer = csv.writer(csv_conn)
        writer.writerow([loss, auc_train_avg, auc_valid_avg, params, self.iter, run_time])

        res = {
            "loss": loss,
            "auc_train": auc_train_avg,
            "auc_valid": auc_valid_avg,
            "params": params,
            "iteration": self.iter,
            "train_time": run_time,
            "status": STATUS_OK,
        }
        print(res)

        return res

    def optimize(self, max_evals):
        self.iter = 0

        space = {
            "learning_rate": hp.choice("learning_rate", [0.01, 0.1]),
            "max_depth": hp.choice("max_depth", [3, 4, 5]),
            "num_leaves": hp.choice("num_leaves", [15, 31, 63]),
            "min_child_samples": hp.choice("min_child_samples", [1, 20, 50]),
        }

        best = fmin(
            fn=self.objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=Trials(),
            max_queue_len=10,
            show_progressbar=True,
            rstate=np.random.default_rng(199198),
        )

        return best


def demo_bo():
    df_woe = data_helper.Data.load("df_woe", prefix=model_config.prefix)
    woe = data_helper.Data.load("woe", prefix=model_config.prefix)
    features = sorted(set(woe.selected_features) - {model_config.pk, model_config.label, "sample_type"})
    df_train = df_woe.loc[df_woe["sample_type"] == "train"]
    df_xtrain, df_ytrain = df_train[features], df_train[model_config.label]

    fp_path = Path(config.ROOT_DIR, "data", model_config.prefix, "bo.csv")
    bo = BO(fp_path)
    bo.load_data(df_xtrain, df_ytrain)
    bo.optimize(80)

    df_cv = pd.read_csv(fp_path)
    df_cv = df_cv.sort_values(by=["auc_valid", "auc_train"], ascending=[False, False]).reset_index(drop=True)
    best_param = df_cv.iloc[0]["params"]
    print(ast.literal_eval(best_param))
    print(df_cv.iloc[0])


def main():
    demo_bo()


if __name__ == "__main__":
    main()
