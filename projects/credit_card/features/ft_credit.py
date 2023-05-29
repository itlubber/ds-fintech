import re

import numpy as np

from projects.credit_card import model_config


class FTCredit:
    @classmethod
    def get_features(cls, df_dim, df_data):
        df_data = cls.get_cum_features(df_data)
        df_res = df_dim.merge(df_data, on=model_config.pk, how="left")

        features = sorted(set(df_res.columns) - {model_config.pk, model_config.label})
        df_res = df_res[[model_config.pk] + features]

        return df_res

    @classmethod
    def get_cum_features(cls, df_data):
        df_data["MAX_PAY_12"] = df_data[["PAY_1", "PAY_2"]].max(axis=1)
        df_data["MAX_PAY_13"] = df_data[["PAY_1", "PAY_2", "PAY_3"]].max(axis=1)
        df_data["MAX_PAY_14"] = df_data[["PAY_1", "PAY_2", "PAY_3", "PAY_4"]].max(axis=1)
        df_data["MAX_PAY_15"] = df_data[["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5"]].max(axis=1)
        df_data["MAX_PAY_16"] = df_data[["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]].max(axis=1)

        df_data["SUM_BILL_12"] = df_data[["BILL_AMT1", "BILL_AMT2"]].sum(axis=1)
        df_data["SUM_BILL_13"] = df_data[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3"]].sum(axis=1)
        df_data["SUM_BILL_14"] = df_data[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4"]].sum(axis=1)
        df_data["SUM_BILL_15"] = df_data[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5"]].sum(axis=1)
        df_data["SUM_BILL_16"] = df_data[
            [
                "BILL_AMT1",
                "BILL_AMT2",
                "BILL_AMT3",
                "BILL_AMT4",
                "BILL_AMT5",
                "BILL_AMT6",
            ]
        ].sum(axis=1)

        df_data["AVG_BILL_12"] = df_data[["BILL_AMT1", "BILL_AMT2"]].mean(axis=1)
        df_data["AVG_BILL_13"] = df_data[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3"]].mean(axis=1)
        df_data["AVG_BILL_14"] = df_data[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4"]].mean(axis=1)
        df_data["AVG_BILL_15"] = df_data[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5"]].mean(axis=1)
        df_data["AVG_BILL_16"] = df_data[
            [
                "BILL_AMT1",
                "BILL_AMT2",
                "BILL_AMT3",
                "BILL_AMT4",
                "BILL_AMT5",
                "BILL_AMT6",
            ]
        ].mean(axis=1)

        df_data["AVG_PAY_AMT_12"] = df_data[["PAY_AMT1", "PAY_AMT2"]].mean(axis=1)
        df_data["AVG_PAY_AMT_13"] = df_data[["PAY_AMT1", "PAY_AMT2", "PAY_AMT3"]].mean(axis=1)
        df_data["AVG_PAY_AMT_14"] = df_data[["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4"]].mean(axis=1)
        df_data["AVG_PAY_AMT_15"] = df_data[["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5"]].mean(axis=1)
        df_data["AVG_PAY_AMT_16"] = df_data[
            ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
        ].mean(axis=1)

        df_data["SUM_PAY_AMT_12"] = df_data[["PAY_AMT1", "PAY_AMT2"]].sum(axis=1)
        df_data["SUM_PAY_AMT_13"] = df_data[["PAY_AMT1", "PAY_AMT2", "PAY_AMT3"]].sum(axis=1)
        df_data["SUM_PAY_AMT_14"] = df_data[["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4"]].sum(axis=1)
        df_data["SUM_PAY_AMT_15"] = df_data[["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5"]].sum(axis=1)
        df_data["SUM_PAY_AMT_16"] = df_data[
            ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
        ].sum(axis=1)

        return df_data
