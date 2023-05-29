from pathlib import Path

import config
from projects.credit_card import model_config
from util import data_helper, report_helper, woe_helper


def run_woe():
    df_sample = data_helper.Data.load("df_sample", prefix=model_config.prefix)
    df_train = df_sample.loc[df_sample["sample_type"] == "train"]

    woe = woe_helper.WOE()
    woe.fit(
        df_train,
        model_config.label,
        method="dt",
        exclude=[model_config.pk, model_config.label, "sample_type"],
    )

    data_helper.Data.dump("woe", woe, prefix=model_config.prefix)


def run_transform():
    df_sample = data_helper.Data.load("df_sample", prefix=model_config.prefix)

    woe = data_helper.Data.load("woe", prefix=model_config.prefix)
    df_woe = woe.transform(df_sample, bin_only=False)
    data_helper.Data.dump("df_woe", df_woe, prefix=model_config.prefix)


def run_bin_valuation():
    df_sample = data_helper.Data.load("df_sample", prefix=model_config.prefix)

    woe = data_helper.Data.load("woe", prefix=model_config.prefix)
    df_bin = woe.transform(df_sample, bin_only=True)

    df_report = report_helper.FTReport.get_report(
        df_bin[woe.selected_features + [model_config.label]], model_config.label
    )
    df_report.to_csv(
        Path(config.ROOT_DIR, "data", model_config.prefix, "feature_report.csv"),
        index=None,
    )


def main():
    # run_woe()
    # run_transform()
    run_bin_valuation()


if __name__ == "__main__":
    main()
