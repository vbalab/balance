import numpy as np
import pandas as pd
from datetime import datetime
from bisect import bisect_left
from pandas.tseries.offsets import MonthEnd

MONTH_MATURITY_MAP_ = {3: 90, 6: 180, 12: 365, 18: 548, 24: 730, 36: 1095}
MATURITY_TO_MONTH_MAP_ = {v: k for k, v in MONTH_MATURITY_MAP_.items()}
MATURITY_ = [v for k, v in MONTH_MATURITY_MAP_.items()]
MONTH_TO_MATURITY_BUCKETS_ = [4, 8, 14, 19, 25]

FTP_RATES_ = ["VTB_ftp_rate_[90d]" for mat in MATURITY_]
MARGIN_ = ["margin_[90d]" for mat in MATURITY_]

SSV_ = ["SSV"]
FOR_ = ["FOR"]

SEGMENT_SPREADS_ = ["priv_spread", "vip_spread"]
OPTIONALITY_SPREADS_ = ["r0s1_spread", "r1s0_spread", "r1s1_spread"]

SBER_RATE_ = ["SBER_max_rate"]
SA_RATE_ = ["SA_rate"]

RENAME_SCENARIO_MAP_ = {"SA_rate": "rate_sa_weighted"}


def month_to_target_maturity(x):
    maturity_num = bisect_left(MONTH_TO_MATURITY_BUCKETS_, x)
    return MATURITY_[maturity_num]


OPTIONALS_ = [(0, 0), (0, 1), (1, 0), (1, 1)]

DEFAULT_SEGMENTS_ = ["mass", "priv", "vip"]
DEFAULT_SEGMENTS_MAP_ = {"mass": 0, "priv": 1, "vip": 2}

NONDEFAULT_SEGMENTS_ = ["mass", "priv", "svip", "bvip"]

PORTFOLIO_COLUMNS_ = [
    "report_month",
    "bucketed_balance",
    "is_vip_or_prv",
    "drop_flg",
    "optional_flg",
    "bucketed_period",
    "open_month",
    "close_month",
    "bucketed_open_rate",
    "report_dt",
    "total_interests",
    "remaining_interests",
    "weight_rate",
    "total_generation",
    "total_generation_cleared",
    "weight_renewal_cnt",
    "weight_renewal_available_flg",
    "weight_close_plan_day",
    "gen_name",
    "SER_d",
    "SER_d_cl",
    "SER_dinamic",
    "SER_dinamic_cl",
    "total_generation_lag1",
    "total_generation_cl_lag1",
    "CUR",
    "report_weight_open_rate_1m",
    "report_weight_open_rate_3m",
    "report_weight_open_rate_6m",
    "report_weight_open_rate_12m",
    "report_weight_open_rate_24m",
    "report_wo_period_weight_open_rate",
    "init_total_generation",
    "row_count",
    "share_period_plan",
    "max_total_generation",
    "max_SER_dinamic",
    "share_buckets_balance",
    "3_med_pr_null",
    "3_med_pr_PENSIONER",
    "3_med_pr_SALARY",
    "3_med_pr_STANDART",
]

BUCKETED_BALANCE_MAP_ = {
    1: "<100k",
    2: "100k-400k",
    3: "400k-1000k",
    4: "1000k-2000k",
    5: ">2000k",
}

SEGMENT_MAP_ = {0: "mass", 1: "priv", 2: "vip"}

# порядок важен
MASS_BALANCE_BUCKETS = [
    "[0_500k)",
    "[500k_1500k)",
    "[1500k_5000k)",
    "[5000k_15000k)",
    "[15000k_inf)",
]

# на момент написания кода совпадают
PRIV_BALANCE_BUCKETS = MASS_BALANCE_BUCKETS  # TODO ???

VIP_BALANCE_BUCKETS = [
    "[0_15kk)",
    "[15kk_30kk)",
    "[30kk_50kk)",
    "[50kk_100kk)",
    "[100kk_200kk)",
    "[200kk_300kk)",
    "[300kk_500kk)",
    "[500kk_inf)",
]


# Пока убрали продукты
SA_PRODUCTS_ = ["general"]  # , 'kopilka', 'safe']


def get_feature_name(
    feature: str,
    segment: str = None,
    repl: int = None,
    sub: int = None,
    maturity: int = None,
) -> str:
    if segment is not None:
        segment_part = f"_[{segment}]"
    else:
        segment_part = ""

    if (repl is not None) and (sub is not None):
        opt_part = f"_[r{repl}s{sub}]"
    elif (repl is None) and (sub is None):
        opt_part = ""
    elif repl is None:
        opt_part = f"_[s{sub}]"
    elif sub is None:
        opt_part = f"_[r{repl}]"
    else:
        raise KeyError("incorrect repl or sub value")

    if maturity is not None:
        mat_part = f"_[{maturity}d]"
    else:
        mat_part = ""

    return f"{feature}{segment_part}{opt_part}{mat_part}"


def get_sa_feature_name(feature, product, segment):
    if product is not None:
        product_part = f"_[{product}]"
    else:
        segment_part = ""
    if segment is not None:
        segment_part = f"_[{segment}]"
    else:
        segment_part = ""

    return f"{feature}{product_part}{segment_part}"


SCENARIO_COLUMNS_ = (
    FTP_RATES_
    + SBER_RATE_
    + ["rate_sa_weighted"]
    + [
        get_feature_name("VTB_weighted_rate", segment, repl, sub, mat)
        for segment in DEFAULT_SEGMENTS_
        for repl, sub in OPTIONALS_
        for mat in MATURITY_
    ]
)
SCENARIO_CAN_BE_EXCLUDED_COLS_ = ["VTB_ftp_rate_[548d]", "VTB_ftp_rate_[1095d]"]
SCENARIO_HARD_COLS_ = [
    col for col in SCENARIO_COLUMNS_ if col not in SCENARIO_CAN_BE_EXCLUDED_COLS_
]


def preprocess_scenario(scenario_df, train_end, horizon):
    scenario_df.loc[:, "mass_spread"] = -scenario_df.loc[:, "vip_spread"]
    scenario_df.loc[:, "priv_spread"] = (
        scenario_df.loc[:, "priv_spread"] - scenario_df.loc[:, "vip_spread"]
    )
    scenario_df.loc[:, "vip_spread"] = 0.0
    scenario_df.loc[:, "r0s0_spread"] = 0.0
    scenario_dates = pd.date_range(
        start=train_end + MonthEnd(1), periods=horizon, freq="M"
    )
    res_df = pd.DataFrame(index=scenario_dates)
    res_df.loc[:, "expected_amount"] = scenario_df["expected_amount"].values
    res_df.loc[scenario_dates, SBER_RATE_] = scenario_df[SBER_RATE_].values
    res_df.loc[scenario_dates, SA_RATE_] = scenario_df[SA_RATE_].values
    res_df.loc[scenario_dates, FTP_RATES_] = scenario_df[FTP_RATES_].values
    # VTB buckets rates
    VTB_buckets_rates = [s for s in list(scenario_df.columns) if "VTB_rate_" in s]
    res_df.loc[scenario_dates, VTB_buckets_rates] = scenario_df[
        VTB_buckets_rates
    ].values
    # default_rates = scenario_df[FTP_RATES_].values - scenario_df[MARGIN_].values - scenario_df[SSV_].values - scenario_df[FOR_].values
    default_rates = (
        scenario_df[FTP_RATES_].values * (1 - scenario_df[FOR_].values / 100)
        - scenario_df[MARGIN_].values
        - scenario_df[SSV_].values
    )
    for segment in DEFAULT_SEGMENTS_:
        for repl, sub in OPTIONALS_:
            rate_names = [
                get_feature_name(
                    "VTB_weighted_rate",
                    segment=segment,
                    repl=repl,
                    sub=sub,
                    maturity=mat,
                )
                for mat in MATURITY_
            ]
            res_df.loc[scenario_dates, rate_names] = (
                default_rates
                + scenario_df[[f"{segment}_spread"]].values
                + scenario_df[[f"r{repl}s{sub}_spread"]].values
            )
            res_df.loc[scenario_dates, rate_names] = np.maximum(
                res_df.loc[scenario_dates, rate_names], 0
            )
    res_df = res_df.rename(columns=RENAME_SCENARIO_MAP_)
    return res_df


def preprocess_scenario_from_file(scenario_df, train_end, horizon):
    for mat in MATURITY_:
        for repl, sub in OPTIONALS_:
            scenario_df.loc[:, f"mass_spread_r{repl}s{sub}_[{mat}d]"] = 0.0
    for mat in MATURITY_:
        for repl, sub in [(0, 0), (0, 1)]:
            scenario_df.loc[:, f"r{repl}s{sub}_spread_[{mat}d]"] = 0.0
    for mat in MATURITY_:
        for repl, sub in [(0, 1)]:
            scenario_df.loc[:, f"margin_r{repl}s{sub}_[{mat}d]"] = scenario_df.loc[
                :, f"margin_r0s0_[{mat}d]"
            ]
            for segment in DEFAULT_SEGMENTS_:
                scenario_df.loc[
                    :, f"{segment}_spread_r{repl}s{sub}_[{mat}d]"
                ] = scenario_df.loc[:, f"{segment}_spread_r0s0_[{mat}d]"]
    scenario_dates = pd.date_range(
        start=train_end + MonthEnd(1), periods=horizon, freq="M"
    )
    res_df = pd.DataFrame(index=scenario_dates)
    res_df.loc[scenario_dates, SBER_RATE_] = scenario_df[SBER_RATE_].values
    res_df.loc[scenario_dates, SA_RATE_] = scenario_df[SA_RATE_].values
    res_df.loc[scenario_dates, FTP_RATES_] = scenario_df[FTP_RATES_].values
    for segment in DEFAULT_SEGMENTS_:
        for repl, sub in OPTIONALS_:
            for mat in MATURITY_:
                default_rates = (
                    scenario_df[f"VTB_ftp_rate_[{mat}d]"].values.reshape(12, 1)
                    * (1 - scenario_df[FOR_].values / 100)
                    - scenario_df[SSV_].values
                )
                rate_name = get_feature_name(
                    "VTB_weighted_rate",
                    segment=segment,
                    repl=repl,
                    sub=sub,
                    maturity=mat,
                )
                res_df.loc[scenario_dates, rate_name] = (
                    default_rates
                    - scenario_df[[f"margin_r{repl}s{sub}_[{mat}d]"]].values
                    - scenario_df[[f"{segment}_spread_r{repl}s{sub}_[{mat}d]"]].values
                    - scenario_df[[f"r{repl}s{sub}_spread_[{mat}d]"]].values
                )
                res_df.loc[scenario_dates, rate_name] = np.maximum(
                    res_df.loc[scenario_dates, rate_name], 0
                )
    res_df = res_df.rename(columns=RENAME_SCENARIO_MAP_)
    return res_df


def preprocess_scenario_from_file_v2(scenario, tech_name, sc_col_name):
    """
    актуальная версия парсера сценариев от Розничного блока

    scenario - загруженный файл сценария
    tech_name - имя колонки с техническими названиями сценария
    sc_col_name - значения сценариев

    TODO: добавить даты по месяцам

    """

    # scenario_dates = pd.date_range(start = train_end + MonthEnd(1), periods=horizon, freq='M')
    scenario_dates = 1
    # Инициализируем, а далее заполняем
    res_df = pd.DataFrame()

    res_df.loc[scenario_dates, "expected_amount"] = np.nan
    res_df.loc[scenario_dates, SBER_RATE_[0]] = scenario[
        scenario[tech_name] == SBER_RATE_[0]
    ][sc_col_name].values[0]
    res_df.loc[scenario_dates, RENAME_SCENARIO_MAP_[SA_RATE_[0]]] = scenario[
        scenario[tech_name] == SA_RATE_[0]
    ][sc_col_name].values[0]
    res_df.loc[scenario_dates, FTP_RATES_] = scenario[
        scenario[tech_name].isin(FTP_RATES_)
    ][sc_col_name].values

    for segment in DEFAULT_SEGMENTS_:
        for repl, sub in OPTIONALS_:
            for mat in MATURITY_:
                default_rates = (
                    scenario[scenario[tech_name] == f"VTB_ftp_rate_[{mat}d]"][
                        sc_col_name
                    ].values[0]
                    * (
                        1
                        - scenario[scenario[tech_name] == FOR_[0]][sc_col_name].values[
                            0
                        ]
                        / 100
                    )
                    - scenario[scenario[tech_name] == SSV_[0]][sc_col_name].values[0]
                )

                rate_name = get_feature_name(
                    "VTB_weighted_rate",
                    segment=segment,
                    repl=repl,
                    sub=sub,
                    maturity=mat,
                )

                if (repl == 0) & (sub == 1):
                    res_df.loc[scenario_dates, rate_name] = 0.01

                else:
                    res_df.loc[scenario_dates, rate_name] = (
                        default_rates
                        - scenario[
                            scenario[tech_name] == f"margin_r{repl}s{sub}_[{mat}d]"
                        ][sc_col_name]
                    ).values[0]

                    if segment != "mass":
                        res_df.loc[scenario_dates, rate_name] = (
                            res_df.loc[scenario_dates, rate_name]
                            - scenario[
                                scenario[tech_name]
                                == f"{segment}_spread_r{repl}s{sub}_[{mat}d]"
                            ][sc_col_name].values[0]
                        )

                    if not ((repl == 0) & (sub == 0)):
                        res_df.loc[scenario_dates, rate_name] = (
                            res_df.loc[scenario_dates, rate_name]
                            - scenario[
                                scenario[tech_name] == f"r{repl}s{sub}_spread_[{mat}d]"
                            ][sc_col_name].values[0]
                        )

                res_df.loc[scenario_dates, rate_name] = np.maximum(
                    res_df.loc[scenario_dates, rate_name], 0.01
                )

                # код написан в сжатые сроки
                res_df1 = res_df.copy()
                for i in range(horizon - 1):
                    res_df1 = res_df1.append(res_df)
                res_df1.index = pd.date_range(
                    start=train_end + MonthEnd(1), periods=horizon, freq="M"
                )

    return res_df1
