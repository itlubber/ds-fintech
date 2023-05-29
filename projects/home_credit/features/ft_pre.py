import numpy as np


class FTPre:
    @classmethod
    def get_features(cls, df_dim, df_pre):
        df_res = df_dim
        df_pre = cls.preprocess(df_pre)

        return df_res

    @classmethod
    def preprocess(cls, df_pre):
        df_pre["DAYS_FIRST_DRAWING"].replace(365243, np.nan, inplace=True)
        df_pre["DAYS_FIRST_DUE"].replace(365243, np.nan, inplace=True)
        df_pre["DAYS_LAST_DUE_1ST_VERSION"].replace(365243, np.nan, inplace=True)
        df_pre["DAYS_LAST_DUE"].replace(365243, np.nan, inplace=True)
        df_pre["DAYS_TERMINATION"].replace(365243, np.nan, inplace=True)

        return df_pre
