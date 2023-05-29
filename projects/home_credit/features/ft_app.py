import numpy as np


class FTApp:
    categorical_features = [
        "CODE_GENDER",
        "WEEKDAY_APPR_PROCESS_START",
        "HOUR_APPR_PROCESS_START",
        "FLAG_MOBIL",
        "FLAG_CONT_MOBILE",
        "FLAG_EMAIL",
        "FLAG_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_EMP_PHONE",
        "FLAG_DOCUMENT_3",
        "FLAG_DOCUMENT_4",
        "FLAG_DOCUMENT_5",
        "FLAG_DOCUMENT_6",
        "FLAG_DOCUMENT_7",
        "FLAG_DOCUMENT_8",
        "FLAG_DOCUMENT_9",
        "FLAG_DOCUMENT_11",
        "FLAG_DOCUMENT_18",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
        "NAME_INCOME_TYPE",
        "NAME_CONTRACT_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "NAME_TYPE_SUITE",
        "LIVE_CITY_NOT_WORK_CITY",
        "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY",
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
        "HOUSETYPE_MODE",
        "EMERGENCYSTATE_MODE",
        "FONDKAPREMONT_MODE",
        "WALLSMATERIAL_MODE",
    ]

    numerical_features = [
        "AMT_ANNUITY",
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL",
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
        "APARTMENTS_AVG",
        "BASEMENTAREA_AVG",
        "COMMONAREA_AVG",
        "ELEVATORS_AVG",
        "ENTRANCES_AVG",
        "LIVINGAPARTMENTS_AVG",
        "LIVINGAREA_AVG",
        "NONLIVINGAPARTMENTS_AVG",
        "NONLIVINGAREA_AVG",
        "FLOORSMAX_AVG",
        "FLOORSMIN_AVG",
        "LANDAREA_AVG",
        "YEARS_BEGINEXPLUATATION_AVG",
        "YEARS_BUILD_AVG",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "DAYS_BIRTH",
        "DAYS_ID_PUBLISH",
        "DAYS_EMPLOYED",
        "OWN_CAR_AGE",
        "DAYS_LAST_PHONE_CHANGE",
        "DAYS_REGISTRATION",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "REGION_POPULATION_RELATIVE",
        "REGION_RATING_CLIENT",
        "TOTALAREA_MODE",
    ]

    @classmethod
    def get_features(cls, df_dim, df_app):
        df_res = df_dim
        df_app = cls.fill_missing_values(df_app)

        df_ft = cls.get_combination_features(df_app)
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        df_ft = cls.get_categorical_features(df_app)
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        df_ft = cls.get_numerical_features(df_app)
        df_res = df_res.merge(df_ft, on=["SK_ID_CURR"], how="left")

        return df_res

    @classmethod
    def fill_missing_values(cls, df_app):
        df_app["CODE_GENDER"].replace("XNA", None, inplace=True)
        df_app["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)
        df_app["DAYS_LAST_PHONE_CHANGE"].replace(0, np.nan, inplace=True)
        df_app["NAME_FAMILY_STATUS"].replace("Unknown", None, inplace=True)
        df_app["ORGANIZATION_TYPE"].replace("XNA", None, inplace=True)
        return df_app

    @classmethod
    def get_combination_features(cls, df_app):
        cols = ["SK_ID_CURR"]

        df_app["ft_app_young_age"] = (df_app["DAYS_BIRTH"] < -14000).astype(int)
        cols.append("ft_app_young_age")
        df_app["ft_app_id_renewal_years"] = (df_app["DAYS_ID_PUBLISH"] - df_app["DAYS_BIRTH"]) / 365
        cols.append("ft_app_id_renewal_years")

        df_app["ft_app_annuity_income_pct"] = df_app["AMT_ANNUITY"] / df_app["AMT_INCOME_TOTAL"]
        cols.append("ft_app_annuity_income_pct")
        df_app["ft_app_credit_income_ratio"] = df_app["AMT_CREDIT"] / df_app["AMT_INCOME_TOTAL"]
        cols.append("ft_app_credit_income_ratio")

        df_app["ft_app_credit_annuity_ratio"] = df_app["AMT_CREDIT"] / df_app["AMT_ANNUITY"]
        cols.append("ft_app_credit_annuity_ratio")
        df_app["ft_app_credit_goods_ratio"] = df_app["AMT_CREDIT"] / df_app["AMT_GOODS_PRICE"]
        cols.append("ft_app_credit_goods_ratio")

        df_app["ft_app_children_ratio"] = df_app["CNT_CHILDREN"] / df_app["CNT_FAM_MEMBERS"]
        cols.append("ft_app_children_ratio")
        df_app["ft_app_income_per_child"] = df_app["AMT_INCOME_TOTAL"] / (1 + df_app["CNT_CHILDREN"])
        cols.append("ft_app_income_per_child")
        df_app["ft_app_income_per_person"] = df_app["AMT_INCOME_TOTAL"] / df_app["CNT_FAM_MEMBERS"]
        cols.append("ft_app_income_per_person")

        df_app["ft_app_days_employed_pct"] = df_app["DAYS_EMPLOYED"] / df_app["DAYS_BIRTH"]
        cols.append("ft_app_days_employed_pct")
        df_app["ft_app_phone_chg_employ_ratio"] = df_app["DAYS_LAST_PHONE_CHANGE"] / df_app["DAYS_EMPLOYED"]
        cols.append("ft_app_phone_chg_employ_ratio")
        df_app["ft_app_short_employment"] = (df_app["DAYS_EMPLOYED"] < -2000).astype(int)
        cols.append("ft_app_short_employment")

        return df_app[cols]

    @classmethod
    def get_categorical_features(cls, df_app):
        df_sub = df_app[["SK_ID_CURR"] + cls.categorical_features]

        for c in cls.categorical_features:
            if c == "SK_ID_CURR":
                continue
            df_sub.loc[:, c] = df_sub[c].apply(lambda x: f"cat_{x}")
            df_sub = df_sub.rename(columns={c: f"ft_app_{c.lower()}"})
        return df_sub

    @classmethod
    def get_numerical_features(cls, df_app):
        df_sub = df_app[["SK_ID_CURR"] + cls.numerical_features]
        for c in cls.numerical_features:
            if c == "SK_ID_CURR":
                continue
            df_sub = df_sub.rename(columns={c: f"ft_app_{c.lower()}"})

        return df_sub
