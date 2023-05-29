import re

import numpy as np


class FTBBalance:
    @classmethod
    def get_features(cls, df_dim, df_bbalance):
        df_res = df_dim
        df_bbalance = cls.preprocess(df_bbalance)

        for period in [360, 720, 1080, 1800, 2520]:
            df_ft = cls.get_feature_per_period(df_dim, df_bbalance, period)
            df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        return df_res

    @classmethod
    def preprocess(cls, df_bbalance):
        df_bbalance.sort_values(["SK_ID_CURR", "DAYS_CREDIT"], ascending=False, inplace=True)
        df_bbalance["no_bureau_balance_history"] = (df_bbalance["STATUS"].isna()).astype(int)
        df_bbalance["bureau_balance_dpd_1plus"] = (df_bbalance["STATUS"].isin(["1", "2", "3", "4", "5"])).astype(int)
        df_bbalance["bureau_balance_dpd_30plus"] = (df_bbalance["STATUS"].isin(["2", "3", "4", "5"])).astype(int)
        return df_bbalance

    @classmethod
    def get_feature_per_period(cls, df_dim, df_bbalance, period):
        df_res = df_dim
        df_sub = df_bbalance[df_bbalance["DAYS_CREDIT"] > (-1) * period]

        df_group = df_sub.groupby(["SK_ID_CURR"])

        df_ft = (
            df_group["no_bureau_balance_history"]
            .agg("sum")
            .reset_index()
            .rename(columns={"no_bureau_balance_history": f"ft_bb_no_history_cnt_{period}d"})
        )
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        df_ft = (
            df_group["bureau_balance_dpd_1plus"]
            .agg("sum")
            .reset_index()
            .rename(columns={"bureau_balance_dpd_1plus": f"ft_bb_bureau_balance_dpd_1plus_cnt_{period}d"})
        )
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        df_ft = (
            df_group["bureau_balance_dpd_30plus"]
            .agg("sum")
            .reset_index()
            .rename(columns={"bureau_balance_dpd_30plus": f"ft_bb_bureau_balance_dpd_30plus_cnt_{period}d"})
        )
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        return df_res
