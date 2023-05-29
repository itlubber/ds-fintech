import numpy as np


class FTCC:
    @classmethod
    def get_features(cls, df_dim, df_cc):
        df_res = df_dim
        df_cc = cls.preprocess(df_cc)

        return df_res

    @classmethod
    def preprocess(cls, df_cc):
        df_cc.loc[df_cc["AMT_DRAWINGS_ATM_CURRENT"] < 0, "AMT_DRAWINGS_ATM_CURRENT"] = np.nan
        df_cc.loc[df_cc["AMT_DRAWINGS_CURRENT"] < 0, "AMT_DRAWINGS_CURRENT"] = np.nan

        return df_cc
