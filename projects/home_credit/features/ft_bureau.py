import re

import numpy as np


class FTBureau:
    @classmethod
    def get_features(cls, df_dim, df_bureau):
        df_res = df_dim
        df_bureau = cls.preprocess(df_bureau)

        for period in [360, 720, 1080, 1800, 2520]:
            df_ft = cls.get_feature_per_period(df_dim, df_bureau, period)
            df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        for period in [360, 720, 1080, 1800, 2520]:
            df_ft = cls.get_pivot_features(df_dim, df_bureau, period)
            df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        return df_res

    @classmethod
    def preprocess(cls, df_bureau):
        df_bureau.sort_values(["SK_ID_CURR", "DAYS_CREDIT"], ascending=False, inplace=True)

        df_bureau.loc[df_bureau["DAYS_CREDIT_ENDDATE"] < -40000, "DAYS_CREDIT_ENDDATE"] = np.nan
        df_bureau.loc[df_bureau["DAYS_CREDIT_UPDATE"] < -40000, "DAYS_CREDIT_UPDATE"] = np.nan
        df_bureau.loc[df_bureau["DAYS_ENDDATE_FACT"] < -40000, "DAYS_ENDDATE_FACT"] = np.nan

        df_bureau["bureau_credit_active_binary"] = (df_bureau["CREDIT_ACTIVE"] != "Closed").astype(int)

        df_bureau["bureau_credit_enddate_binary"] = (df_bureau["DAYS_CREDIT_ENDDATE"] > 0).astype(int)

        df_bureau["bureau_credit_type_consumer"] = (df_bureau["CREDIT_TYPE"] == "Consumer credit").astype(int)
        df_bureau["bureau_credit_type_credit_card"] = (df_bureau["CREDIT_TYPE"] == "Credit card").astype(int)
        df_bureau["bureau_credit_type_car"] = (df_bureau["CREDIT_TYPE"] == "Car loan").astype(int)
        df_bureau["bureau_credit_type_mortgage"] = (df_bureau["CREDIT_TYPE"] == "Mortgage").astype(int)
        df_bureau["bureau_credit_type_other"] = (
            ~(df_bureau["CREDIT_TYPE"].isin(["Consumer credit", "Car loan", "Mortgage", "Credit card"]))
        ).astype(int)

        df_bureau["bureau_unusual_currency"] = (~(df_bureau["CREDIT_CURRENCY"] == "currency 1")).astype(int)

        df_bureau["days_credit_diff"] = df_bureau["DAYS_CREDIT"].diff().replace(np.nan, 0)

        df_bureau["debt_credit_ratio"] = df_bureau["AMT_CREDIT_SUM_DEBT"] / df_bureau["AMT_CREDIT_SUM"]

        return df_bureau

    @classmethod
    def get_feature_per_period(cls, df_dim, df_bureau, period):
        df_res = df_dim
        df_sub = df_bureau[df_bureau["DAYS_CREDIT"] > (-1) * period]
        df_group = df_sub.groupby(["SK_ID_CURR"])

        df_ft = (
            df_group["DAYS_CREDIT"]
            .agg("count")
            .reset_index()
            .rename(columns={"DAYS_CREDIT": f"ft_bureau_query_cnt_{period}d"})
        )
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        df_ft = (
            df_group["CREDIT_TYPE"]
            .agg("nunique")
            .reset_index()
            .rename(columns={"CREDIT_TYPE": f"ft_bureau_unique_credit_type_cnt_{period}d"})
        )
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        df_ft = (
            df_group["debt_credit_ratio"]
            .agg("mean")
            .reset_index()
            .rename(columns={"debt_credit_ratio": f"ft_bureau_avg_debt_credit_ratio_{period}d"})
        )
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        df_ft = (
            df_group["debt_credit_ratio"]
            .agg("max")
            .reset_index()
            .rename(columns={"debt_credit_ratio": f"ft_bureau_max_debt_credit_ratio_{period}d"})
        )
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        df_ft = (
            df_group["debt_credit_ratio"]
            .agg("last")
            .reset_index()
            .rename(columns={"debt_credit_ratio": f"ft_bureau_recent_debt_credit_ratio_{period}d"})
        )
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        df_ft = (
            df_group["AMT_CREDIT_SUM_DEBT"]
            .agg("sum")
            .reset_index()
            .rename(columns={"AMT_CREDIT_SUM_DEBT": f"ft_bureau_total_credit_debt_amt_{period}d"})
        )
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        df_ft = (
            df_group["AMT_CREDIT_SUM"]
            .agg("sum")
            .reset_index()
            .rename(columns={"AMT_CREDIT_SUM": f"ft_bureau_total_credit_amt_{period}d"})
        )
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        df_ft = (
            df_group["AMT_CREDIT_SUM_OVERDUE"]
            .agg("last")
            .reset_index()
            .rename(columns={"AMT_CREDIT_SUM_OVERDUE": f"ft_bureau_total_overdue_credit_amt_{period}d"})
        )
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        df_ft = (
            df_group["CNT_CREDIT_PROLONG"]
            .agg("last")
            .reset_index()
            .rename(columns={"CNT_CREDIT_PROLONG": f"ft_bureau_total_prolong_credit_amt_{period}d"})
        )
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        df_ft = (
            df_group["bureau_credit_enddate_binary"]
            .agg("last")
            .reset_index()
            .rename(columns={"bureau_credit_enddate_binary": f"ft_bureau_credit_enddate_pct_{period}d"})
        )
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        return df_res

    @classmethod
    def get_pivot_features(cls, df_dim, df_bureau, period):
        df_res = df_dim
        df_sub = df_bureau[df_bureau["DAYS_CREDIT"] > (-1) * period]

        lst_cat = list(df_bureau["CREDIT_TYPE"].unique())

        dict_rename = dict()
        for k in lst_cat:
            v = re.sub(r"[()]", "", k.lower())
            v = re.sub(r"\s+", "_", v)
            val = f"ft_bureau_cat_{v}_credit_amt_{period}d"
            dict_rename[k] = val

        df_ft = df_sub[["SK_ID_CURR", "CREDIT_TYPE", "AMT_CREDIT_SUM"]].pivot_table(
            index="SK_ID_CURR",
            columns=["CREDIT_TYPE"],
            values="AMT_CREDIT_SUM",
            aggfunc="sum",
        )
        df_ft = df_ft.rename(columns=dict_rename)
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        dict_rename = dict()
        for k in lst_cat:
            v = re.sub(r"[()]", "", k.lower())
            v = re.sub(r"\s+", "_", v)
            val = f"ft_bureau_cat_{v}_credit_debt_amt_{period}d"
            dict_rename[k] = val

        df_ft = df_bureau[["SK_ID_CURR", "CREDIT_TYPE", "AMT_CREDIT_SUM_DEBT"]].pivot_table(
            index="SK_ID_CURR",
            columns=["CREDIT_TYPE"],
            values="AMT_CREDIT_SUM_DEBT",
            aggfunc="sum",
        )
        df_ft = df_ft.rename(columns=dict_rename)
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        for k in lst_cat:
            v = re.sub(r"[()]", "", k.lower())
            v = re.sub(r"\s+", "_", v)

            pre = f"ft_bureau_cat_{v}"
            name = f"{pre}_debt_credit_ratio_{period}d"
            nom = f"{pre}_credit_debt_amt_{period}d"
            denom = f"{pre}_credit_amt_{period}d"
            if nom in df_res and denom in df_res:
                df_res[name] = df_res[nom] / df_res[denom]

        return df_res
