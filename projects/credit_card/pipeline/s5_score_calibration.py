from projects.credit_card import model_config
from util import data_helper, report_helper, score_helper, scorebin_helper


def run_bm_score_calibration():
    df_score = data_helper.Data.load("df_bm_result", prefix=model_config.prefix)

    sc = score_helper.Score()
    sc.fit(df_score["bm_score"], df_score[model_config.label])
    df_score["bm_credit_score"] = sc.transform(df_score["bm_score"])

    df_score = scorebin_helper.ScoreBin.bin_megascore(df_score, "bm_credit_score")
    df_report = report_helper.ScoreReport.get_report(df_score, "bin", model_config.label)
    print(df_report)

    df_score = scorebin_helper.ScoreBin.bin_quantile(df_score, "bm_credit_score", model_config.label)
    df_report = report_helper.ScoreReport.get_report(df_score, "bin", model_config.label)
    print(df_report)


def run_lgbm_score_calibration():
    df_score = data_helper.Data.load("df_lgbm_result", prefix=model_config.prefix)

    sc = score_helper.Score()
    sc.fit(df_score["lgbm_score"], df_score[model_config.label])
    df_score["lgbm_credit_score"] = sc.transform(df_score["lgbm_score"])

    df_score = scorebin_helper.ScoreBin.bin_megascore(df_score, "lgbm_credit_score")
    df_report = report_helper.ScoreReport.get_report(df_score, "bin", model_config.label)
    print(df_report)

    df_score = scorebin_helper.ScoreBin.bin_quantile(df_score, "lgbm_credit_score", model_config.label)
    df_report = report_helper.ScoreReport.get_report(df_score, "bin", model_config.label)
    print(df_report)


def main():
    run_bm_score_calibration()
    run_lgbm_score_calibration()


if __name__ == "__main__":
    main()
