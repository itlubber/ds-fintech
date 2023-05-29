from projects.credit_card import model_config
from util import bm_helper, data_helper, lgbm_helper, report_helper


def run_benchmark():
    df_woe = data_helper.Data.load("df_woe", prefix=model_config.prefix)
    woe = data_helper.Data.load("woe", prefix=model_config.prefix)

    features = sorted(set(woe.selected_features) - {model_config.pk, model_config.label, "sample_type"})
    df_train = df_woe.loc[df_woe["sample_type"] == "train"]
    df_xtrain, df_ytrain = df_train[features], df_train[model_config.label]

    model = bm_helper.Logit()
    model.select_and_fit(df_xtrain, df_ytrain)
    data_helper.Data.dump("bm_model", model, prefix=model_config.prefix)
    print(model.get_importance())

    df_woe["bm_score"] = model.predict(df_woe)
    data_helper.Data.dump("df_bm_result", df_woe, prefix=model_config.prefix)

    df_report = report_helper.ModelReport.get_report(df_woe, "sample_type", "bm_score", model_config.label)
    print(df_report)


def main():
    run_benchmark()


if __name__ == "__main__":
    main()
