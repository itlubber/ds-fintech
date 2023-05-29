from pathlib import Path

import numpy as np
import pandas as pd

import config
from projects.home_credit import model_config
from projects.home_credit.features import ft_app, ft_bureau, ft_bureau_balance, ft_install
from util import data_helper, db_helper


def get_sample_with_label():
    db_path = Path(config.ROOT_DIR, "data", "home_credit_default_risk.db")
    db_helper.SQLite.get_conn(db_path)
    sql = """
        SELECT * FROM application_train
    """

    df_app = db_helper.SQLite.query(sql)

    df_dim = pd.DataFrame({model_config.pk: df_app[model_config.pk].unique()})
    df_label = data_helper.Data.train_test_split(df_app, model_config.pk, model_config.label)

    data_helper.Data.dump("df_dim", df_dim, prefix="home_credit")
    data_helper.Data.dump("df_label", df_label, prefix="home_credit")


def get_app_features():
    db_path = Path(config.ROOT_DIR, "data", "home_credit_default_risk.db")
    db_helper.SQLite.get_conn(db_path)
    sql = """
        SELECT * FROM application_train
    """

    df_ft = db_helper.SQLite.query(sql)

    df_dim = data_helper.Data.load("df_dim", prefix="home_credit")
    df_ft = ft_app.FTApp.get_features(df_dim, df_ft)
    data_helper.Data.dump("df_app_ft", df_ft, prefix="home_credit")


def get_install_features():
    db_path = Path(config.ROOT_DIR, "data", "home_credit_default_risk.db")
    db_helper.SQLite.get_conn(db_path)
    sql = """
            SELECT * FROM installments_payments
    """

    df_ft = db_helper.SQLite.query(sql)

    df_dim = data_helper.Data.load("df_dim", prefix="home_credit")
    df_ft = ft_install.FTInstall.get_features(df_dim, df_ft)
    data_helper.Data.dump("df_install_ft", df_ft, prefix="home_credit")


def get_bureau_features():
    db_path = Path(config.ROOT_DIR, "data", "home_credit_default_risk.db")
    db_helper.SQLite.get_conn(db_path)
    sql = """
            SELECT * FROM bureau
    """

    df_ft = db_helper.SQLite.query(sql)

    df_dim = data_helper.Data.load("df_dim", prefix="home_credit")
    df_ft = ft_bureau.FTBureau.get_features(df_dim, df_ft)
    data_helper.Data.dump("df_bureau_ft", df_ft, prefix="home_credit")


def get_bbalance_features():
    db_path = Path(config.ROOT_DIR, "data", "home_credit_default_risk.db")
    db_helper.SQLite.get_conn(db_path)
    sql = """
        WITH 
        balance_tab AS (
            SELECT SK_ID_BUREAU, MAX(MONTHS_BALANCE) AS MONTHS_BALANCE
            FROM bureau_balance
            GROUP BY SK_ID_BUREAU
        ),
        recent_balance_tab AS (
            SELECT U.SK_ID_BUREAU, U.MONTHS_BALANCE, BB.STATUS
            FROM balance_tab U
            LEFT JOIN bureau_balance BB
            ON U.SK_ID_BUREAU=BB.SK_ID_BUREAU AND U.MONTHS_BALANCE = BB.MONTHS_BALANCE
        )

        SELECT B.SK_ID_CURR, B.SK_ID_BUREAU, B.DAYS_CREDIT, 
               RB.MONTHS_BALANCE, RB.STATUS
        FROM bureau B
        LEFT JOIN recent_balance_tab RB
        ON B.SK_ID_BUREAU=RB.SK_ID_BUREAU
    """

    df_ft = db_helper.SQLite.query(sql)

    df_dim = data_helper.Data.load("df_dim", prefix="home_credit")
    df_ft = ft_bureau_balance.FTBBalance.get_features(df_dim, df_ft)
    data_helper.Data.dump("df_bbalance_ft", df_ft, prefix="home_credit")


def combine():
    df_label = data_helper.Data.load("df_label", prefix="home_credit")
    df_app_ft = data_helper.Data.load("df_app_ft", prefix="home_credit")
    df_install_ft = data_helper.Data.load("df_install_ft", prefix="home_credit")
    df_bureau_ft = data_helper.Data.load("df_bureau_ft", prefix="home_credit")
    df_bbalance_ft = data_helper.Data.load("df_bbalance_ft", prefix="home_credit")

    df_sample = data_helper.Data.combine_sample(
        model_config.pk,
        df_label,
        [df_app_ft, df_install_ft, df_bureau_ft, df_bbalance_ft],
    )
    data_helper.Data.dump("df_sample", df_sample, prefix="home_credit")


def main():
    get_sample_with_label()
    get_app_features()
    get_install_features()
    get_bureau_features()
    get_bbalance_features()
    combine()


if __name__ == "__main__":
    main()
