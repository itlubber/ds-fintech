class FTInstall:
    @classmethod
    def get_features(cls, df_dim, df_install):
        df_install["ft_instal_dpd"] = df_install["DAYS_ENTRY_PAYMENT"] - df_install["DAYS_INSTALMENT"]
        df_install["ft_instal_dpd"] = df_install["ft_instal_dpd"].apply(lambda x: max(0, x))
        df_install["ft_instal_has_dpd"] = (df_install["ft_instal_dpd"] > 0).astype(int)
        df_install["ft_instal_paid_diff_amt"] = df_install["AMT_PAYMENT"] - df_install["AMT_INSTALMENT"]
        df_install["ft_instal_is_over_paid"] = (df_install["ft_instal_paid_diff_amt"] > 0).astype(int)
        df_install["ft_instal_is_under_paid"] = (df_install["ft_instal_paid_diff_amt"] < 0).astype(int)

        df_res = df_dim
        for period in [30, 60, 90, 120, 360, 720]:
            df_ft = cls.get_feature_per_period(df_dim, df_install, period)
            df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        return df_res

    @classmethod
    def get_feature_per_period(cls, df_dim, df_install, period):
        df_res = df_dim
        df_sub = df_install[df_install["DAYS_INSTALMENT"] > (-1) * period]
        df_group = df_sub.groupby(["SK_ID_CURR"])

        df_ft = (
            df_group["ft_instal_dpd"]
            .agg("max")
            .reset_index()
            .rename(columns={"ft_instal_dpd": f"ft_instal_max_dpd_{period}d"})
        )
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        df_ft = (
            df_group["ft_instal_has_dpd"]
            .agg("sum")
            .reset_index()
            .rename(columns={"ft_instal_has_dpd": f"ft_instal_dpd_cnt_{period}d"})
        )
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        df_ft = (
            df_group["ft_instal_is_over_paid"]
            .agg("sum")
            .reset_index()
            .rename(columns={"ft_instal_is_over_paid": f"ft_instal_over_paid_cnt_{period}d"})
        )
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        df_ft = (
            df_group["ft_instal_is_under_paid"]
            .agg("sum")
            .reset_index()
            .rename(columns={"ft_instal_is_under_paid": f"ft_instal_under_paid_cnt_{period}d"})
        )
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        df_ft = (
            df_group["AMT_PAYMENT"]
            .agg("sum")
            .reset_index()
            .rename(columns={"AMT_PAYMENT": f"ft_instal_total_paid_amt_{period}d"})
        )
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        df_ft = (
            df_group["AMT_PAYMENT"]
            .agg("mean")
            .reset_index()
            .rename(columns={"AMT_PAYMENT": f"ft_instal_avg_paid_amt_{period}d"})
        )
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        df_ft = (
            df_group["AMT_PAYMENT"]
            .agg("max")
            .reset_index()
            .rename(columns={"AMT_PAYMENT": f"ft_instal_max_paid_amt_{period}d"})
        )
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        df_ft = (
            df_group["AMT_PAYMENT"]
            .agg("min")
            .reset_index()
            .rename(columns={"AMT_PAYMENT": f"ft_instal_min_paid_amt_{period}d"})
        )
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        return df_res
