from projects.home_credit import model_config
from util import bm_helper, data_helper, lgbm_helper, report_helper


def run_lgbm():
    df_woe = data_helper.Data.load("df_woe", prefix="home_credit")
    woe = data_helper.Data.load("woe", prefix="home_credit")

    features = sorted(set(woe.selected_features) - {model_config.pk, model_config.label, "sample_type"})
    df_train = df_woe.loc[df_woe["sample_type"] == "train"]
    df_valid = df_woe.loc[df_woe["sample_type"] == "test"]
    df_xtrain, df_ytrain = df_train[features], df_train[model_config.label]
    df_xvalid, df_yvalid = df_valid[features], df_valid[model_config.label]

    model = lgbm_helper.LightGBM()
    model.select_and_fit(df_xtrain, df_ytrain, df_xvalid=df_xvalid, df_yvalid=df_yvalid, n_estimators=500)
    data_helper.Data.dump("lgbm_model", model, prefix="home_credit")
    print(model.get_importance())

    df_woe["lgbm_score"] = model.predict(df_woe)
    data_helper.Data.dump("df_lgbm_result", df_woe, prefix="home_credit")

    df_report = report_helper.ModelReport.get_report(df_woe, "sample_type", "lgbm_score", model_config.label)
    print(df_report)


def main():
    run_lgbm()


if __name__ == "__main__":
    main()
