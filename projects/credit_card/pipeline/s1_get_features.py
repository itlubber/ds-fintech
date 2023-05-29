from pathlib import Path

import numpy as np
import pandas as pd

import config
from projects.credit_card import model_config
from projects.credit_card.features import ft_credit
from util import data_helper, db_helper


def get_sample_with_label():
    db_path = Path(config.ROOT_DIR, "data", "credit_card.db")
    db_helper.SQLite.get_conn(db_path)
    sql = """
            SELECT * FROM credit_card
        """
    df_data = db_helper.SQLite.query(sql)

    df_dim = pd.DataFrame({model_config.pk: df_data[model_config.pk].unique()})
    df_label = data_helper.Data.train_test_split(df_data, model_config.pk, model_config.label)

    data_helper.Data.dump("df_dim", df_dim, prefix=model_config.prefix)
    data_helper.Data.dump("df_label", df_label, prefix=model_config.prefix)


def get_features():
    db_path = Path(config.ROOT_DIR, "data", "credit_card.db")
    db_helper.SQLite.get_conn(db_path)
    sql = """
            SELECT * FROM credit_card
        """

    df_data = db_helper.SQLite.query(sql)

    df_dim = data_helper.Data.load("df_dim", prefix=model_config.prefix)
    df_ft = ft_credit.FTCredit.get_features(df_dim, df_data)
    data_helper.Data.dump("df_credit_ft", df_ft, prefix=model_config.prefix)


def combine():
    df_label = data_helper.Data.load("df_label", prefix=model_config.prefix)
    df_credit_ft = data_helper.Data.load("df_credit_ft", prefix=model_config.prefix)

    df_sample = data_helper.Data.combine_sample(model_config.pk, df_label, [df_credit_ft])
    data_helper.Data.dump("df_sample", df_sample, prefix=model_config.prefix)


def main():
    get_sample_with_label()
    get_features()
    combine()


if __name__ == "__main__":
    main()
