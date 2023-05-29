import re
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from dateutil import relativedelta
from sklearn.model_selection import GridSearchCV

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

import config
from projects.credit_card import model_config
from util import data_helper


def demo_grid_search():
    df_woe = data_helper.Data.load("df_woe", prefix=model_config.prefix)
    woe = data_helper.Data.load("woe", prefix=model_config.prefix)
    features = sorted(set(woe.selected_features) - {model_config.pk, model_config.label, "sample_type"})
    df_train = df_woe.loc[df_woe["sample_type"] == "train"]
    df_xtrain, df_ytrain = df_train[features], df_train[model_config.label]

    params = {
        "learning_rate": 0.1,
        "max_depth": 3,
        "num_leaves": 7,
        "min_child_samples": 800,
        "subsample": 1,
        "subsample_freq": 0,
        "colsample_bytree": 1,
        "reg_alpha": 157,
        "reg_lambda": 500,
    }

    model = lgb.LGBMClassifier(
        **params,
        n_estimators=200,
        objective="cross_entropy",
        class_weight="balanced",
        importance_type="gain",
        boosting_type="gbdt",
        silent=True,
        n_jobs=8,
        random_state=19910908
    )

    # param_grid = {
    #     'learning_rate': [0.01, 0.1],
    #     'max_depth': [3, 4, 5],
    #     'num_leaves': [15, 31, 63],
    #     'min_child_samples': [1, 20, 50],
    #     'subsample': [0.8, 1.0],
    #     'colsample_bytree': [0.8, 1.0]
    # }

    param_grid = {
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 4, 5],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [1, 20, 50],
    }

    grid = GridSearchCV(
        model,
        param_grid,
        verbose=3,
        cv=3,
        scoring={"AUC": "roc_auc"},
        n_jobs=1,
        refit="AUC",
    )
    result = grid.fit(df_xtrain, df_ytrain)

    print(result.best_score_)
    print(result.best_params_)

    df_cv = pd.DataFrame(result.cv_results_["params"])
    df_cv["mean_test_AUC"] = result.cv_results_["mean_test_AUC"]
    print(df_cv)

    fp_path = Path(config.ROOT_DIR, "data", model_config.prefix, "grid.csv")
    df_cv.to_csv(fp_path, index=None)


def main():
    demo_grid_search()


if __name__ == "__main__":
    main()
