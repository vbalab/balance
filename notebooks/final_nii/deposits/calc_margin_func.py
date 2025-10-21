import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
from bisect import bisect_left


SEGMENT_MAP_ = {0: "mass", 1: "priv", 2: "vip"}


MATURITY_ = [90, 180, 365, 548, 730, 1095]
MONTH_MATURITY_MAP_ = {3: 90, 6: 180, 12: 365, 18: 548, 24: 730, 36: 1095}
MATURITY_TO_MONTH_MAP_ = {value: key for key, value in MONTH_MATURITY_MAP_.items()}
MONTH_TO_MATURITY_BUCKETS_ = [4, 8, 14, 19, 25]


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
]

BUCKETED_BALANCE_MAP_ = {
    1: "<100k",
    2: "100k-400k",
    3: "400k-1000k",
    4: "1000k-2000k",
    5: ">2000k",
}


def month_to_target_maturity(x):
    maturity_num = bisect_left(MONTH_TO_MATURITY_BUCKETS_, x)
    return MATURITY_[maturity_num]


## факт портфель


def _add_cols_to_port(port):
    port = port.copy()
    port.loc[:, "replenishable_flg"] = port["optional_flg"].isin([2, 3]).astype(int)
    port.loc[:, "subtraction_flg"] = port["optional_flg"].isin([1, 3]).astype(int)
    port.loc[:, "segment"] = port["is_vip_or_prv"].apply(lambda x: SEGMENT_MAP_[x])
    port.loc[:, "month_maturity"] = (port["bucketed_period"] - 1).astype(int)
    port.loc[:, "target_maturity_days"] = port.month_maturity.apply(
        month_to_target_maturity
    )
    return port


def _portfolio_result(port):
    port["bucketed_balance_nm"] = port.bucketed_balance.apply(
        lambda x: BUCKETED_BALANCE_MAP_[x] if x in BUCKETED_BALANCE_MAP_ else "<100k"
    )
    port.loc[:, "renewal_cnt"] = port.weight_renewal_cnt.round()
    port.loc[:, "operations_in_month"] = np.where(
        port.optional_flg > 0, port.SER_d_cl, 0
    )
    port.loc[:, "early_withdrawal_in_month"] = np.where(
        port.optional_flg == 0, port.SER_d_cl, 0
    )
    port.loc[:, "balance"] = port.loc[:, "total_generation"]

    # port = port[port['report_dt']>self._forecast_context.portfolio_dt+MonthEnd(-1)]
    return port.sort_values(
        by=["report_dt", "segment", "replenishable_flg", "subtraction_flg", "balance"],
        ascending=[True, True, True, True, False],
        ignore_index=True,
    )


def prep_fact_port(port_res):

    BUCKETED_BALANCE_MAP_ = {
        1: "<100k",
        2: "100k-400k",
        3: "400k-1000k",
        4: "1000k-2000k",
        5: ">2000k",
    }

    portfolio_result_cols = [
        "report_dt",
        "segment",
        "replenishable_flg",
        "subtraction_flg",
        "month_maturity",
        "target_maturity_days",
        "bucketed_balance_nm",
        "bucketed_balance",
        "open_month",
        "close_month",
        "weight_rate",
        "balance",
        "renewal_cnt",
        "operations_in_month",
        "early_withdrawal_in_month",
        "gen_name",
    ]

    port_res = _add_cols_to_port(port_res)
    port_res = _portfolio_result(port_res)

    return port_res


def read_fact_data(month):

    i = month
    port = pd.read_csv(
        f"/home/vtb70186744/dynbalance/data/portfolio_data/portfolio_2023-0{i}.csv",
        parse_dates=True,
    )
    port["report_dt"] = pd.to_datetime(port["report_dt"])

    port = prep_fact_port(port)

    return port


## для прогнозов


def get_spreads():

    spreads = pd.read_excel(
        "/home/vtb70186744/dynbalance/examples/net_perc_income/spreads_deposits.xlsx"
    )

    drop_cols = [
        "Валюта",
        "Комментарий",
        "Обозначение инструмента",
        "Обозначение базиса плавающей ставки",
        "4 года (1461 день)",
        "5 лет (1830 дней)",
        "2 недели (14 дней)",
        "3 недели (21 день)",
        "1 месяц (31 день)",
        "2 месяца (61 день)",
        "9 месяцев (271 день)",
    ]

    spreads.drop(columns=drop_cols, inplace=True)

    spreads_maturity_dict = {
        "3 месяца (91 день)": "90_spread",
        "6 месяцев (181 день)": "180_spread",
        "1 год (365 дней)": "365_spread",
        "1,5 года (548 дней)": "548_spread",
        "2 года (731 день)": "730_spread",
        "3 года (1095 дней)": "1095_spread",
    }
    spreads.rename(columns=spreads_maturity_dict, inplace=True)

    spreads.loc[:, "replenishable_flg"] = (
        spreads["Инструмент"]
        .isin(
            [
                "Спред по депозитам ФЛ с правом пополнения",
                "Спред по депозитам ФЛ с правом пополнения и досрочного частичного снятия до суммы неснижаемого остатка",
            ]
        )
        .astype(int)
    )
    spreads.loc[:, "subtraction_flg"] = (
        spreads["Инструмент"]
        .isin(
            [
                "Спред по депозитам ФЛ с правом пополнения и досрочного частичного снятия до суммы неснижаемого остатка"
            ]
        )
        .astype(int)
    )

    ## ПОДГОТОВИМ ДЛЯ ЗАПИСИ
    spreads_tmp = spreads.rename(columns={"Дата": "date"})

    spreads_tmp_s0 = spreads_tmp[spreads_tmp["subtraction_flg"] == 0]
    spreads_tmp_s1 = spreads_tmp[spreads_tmp["subtraction_flg"] == 1]

    dates = pd.date_range(start="2005-01-01", end="2023-09-01")

    dates = pd.DataFrame(dates)
    dates.rename(columns={0: "date"}, inplace=True)
    spreads_tmp_s0 = (
        dates.merge(spreads_tmp_s0, on="date", how="left")
        .sort_values(by="date", ascending=False)
        .fillna(method="bfill")
        .dropna()
    )

    del spreads_tmp_s0["Инструмент"]
    spreads_tmp_s1 = (
        dates.merge(spreads_tmp_s1, on="date", how="left")
        .sort_values(by="date", ascending=False)
        .fillna(method="bfill")
        .dropna()
    )

    del spreads_tmp_s1["Инструмент"]

    spreads_tmp_final = spreads_tmp_s0.append(spreads_tmp_s1)

    spreads["open_month"] = spreads["Дата"].apply(lambda x: str(x)[:7])

    spreads = spreads[
        [
            "90_spread",
            "180_spread",
            "365_spread",
            "548_spread",
            "730_spread",
            "1095_spread",
            "replenishable_flg",
            "subtraction_flg",
            "open_month",
        ]
    ]

    years = range(2018, 2024, 1)
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

    dates_list = []
    for year in years:
        for month in months:
            date = str(year) + "-" + month
            dates_list.append(date)

    spreads.index = pd.MultiIndex.from_frame(spreads[["subtraction_flg", "open_month"]])
    spreads = spreads.drop(columns=["subtraction_flg", "open_month"])
    cols_sp = list(spreads.columns)

    for i, date in enumerate(dates_list):
        for sub in [0, 1]:
            if (sub, date) not in spreads.index:
                # print((sub, date))
                spreads.loc[(sub, date), cols_sp] = (
                    spreads.loc[sub, dates_list[i - 1]]
                    .reset_index()[cols_sp]
                    .values[0]
                    .tolist()
                )

    spreads = spreads.reset_index()

    spreads["replenishable_flg"] = spreads["replenishable_flg"].astype(int)

    spreads = spreads.drop_duplicates(
        subset=["open_month", "subtraction_flg", "replenishable_flg"]
    )

    return spreads


def get_ftp():

    ftp_table = pd.read_csv(
        "/home/vtb70186744/dynbalance/examples/net_perc_income/ftp.csv"
    )

    # Корректируем ставку за февраль 2022 в связи с резкими притоками
    ind = ftp_table[
        (ftp_table["report_year"] == 2022) & (ftp_table["report_month"] == 2)
    ].index
    cols = ftp_table.columns[2:]

    w1 = 5
    w2 = 1

    new_ftp = (
        (
            ftp_table[
                (ftp_table["report_year"] == 2022) & (ftp_table["report_month"] == 2)
            ][cols]
            * w1
        ).values
        + (
            ftp_table[
                (ftp_table["report_year"] == 2022) & (ftp_table["report_month"] == 3)
            ][cols]
            * w2
        ).values
    ) / (w1 + w2)

    ftp_table.at[ind, cols] = new_ftp
    ftp_table.at[ind, "report_month"] = 2

    ftp_table["report_month"][ftp_table["report_month"] <= 9] = ftp_table[
        "report_month"
    ][ftp_table["report_month"] <= 9].apply(lambda x: "0" + str(x))
    ftp_table["report_month"] = ftp_table["report_month"].astype(str)
    ftp_table["report_year"] = ftp_table["report_year"].astype(str)
    ftp_table["open_month"] = ftp_table["report_year"] + "-" + ftp_table["report_month"]

    del ftp_table["report_month"]
    del ftp_table["report_year"]

    return ftp_table


def get_for():

    d = {
        "2023-01": [3],
        "2023-02": [3],
        "2023-03": [4],
        "2023-04": [4],
        "2023-05": [4],
        "2023-06": [4.5],
        "2023-07": [4.5],
    }

    FOR = pd.DataFrame(data=d, index=None).T

    FOR = FOR.reset_index().rename(columns={"index": "report_month", 0: "FOR"})

    return FOR


def correct_mat(df):

    portfolio_res = df.copy()
    # корректируем месяца

    maturity_dict = {90: 3, 180: 6, 365: 12, 548: 18, 730: 24, 1095: 36}

    portfolio_res["target_maturity_months"] = portfolio_res[
        "target_maturity_days"
    ].replace(maturity_dict)

    # Корректируем дату открытия относительно даты закрытия для депозитов с пролонгацией

    portfolio_res["close_month_dt"] = pd.to_datetime(portfolio_res["close_month"])
    portfolio_res["open_month_correct"] = pd.to_datetime(portfolio_res["open_month"])

    for mat in portfolio_res["target_maturity_months"].unique():

        portfolio_res["open_month_correct"][
            (portfolio_res["renewal_cnt"] > 0)
            & (portfolio_res["target_maturity_months"] == mat)
        ] = portfolio_res["close_month_dt"][
            (portfolio_res["renewal_cnt"] > 0)
            & (portfolio_res["target_maturity_months"] == mat)
        ] + MonthEnd(
            -mat
        )

        portfolio_res["open_month"][
            (portfolio_res["renewal_cnt"] > 0)
            & (portfolio_res["target_maturity_months"] == mat)
        ] = portfolio_res["open_month_correct"][
            (portfolio_res["renewal_cnt"] > 0)
            & (portfolio_res["target_maturity_months"] == mat)
        ].apply(
            lambda x: str(x)[:7]
        )

    return portfolio_res


def margin_calc(df):
    portfolio_res = df.copy()

    portfolio_res = correct_mat(portfolio_res)
    spreads = get_spreads()
    portfolio_res = portfolio_res.merge(
        spreads, on=["replenishable_flg", "subtraction_flg", "open_month"], how="left"
    )
    portfolio_res["spreads_mat"] = np.nan

    for day in portfolio_res["target_maturity_days"].unique():

        # выбираем колонку
        col = f"{day}_spread"

        portfolio_res["spreads_mat"][portfolio_res["target_maturity_days"] == day] = (
            portfolio_res[col][portfolio_res["target_maturity_days"] == day]
        )

    portfolio_res["spreads_mat"] = portfolio_res["spreads_mat"].fillna(0) * 100
    ftp_table = get_ftp()

    # дополнить этот момент
    # для новых вкладов лучше брать новые ставки

    portfolio_res["open_month"][portfolio_res["open_month"] > "2023-09"] = "2023-09"

    portfolio_res = portfolio_res.merge(ftp_table, on="open_month", how="left")

    portfolio_res["ftp"] = np.nan

    for day in portfolio_res["target_maturity_days"].unique():

        # выбираем колонку
        col = f"vtb_[{day}d]_ftp_rate"

        portfolio_res["ftp"][portfolio_res["target_maturity_days"] == day] = (
            portfolio_res[col][portfolio_res["target_maturity_days"] == day]
        )

    FOR = get_for()
    portfolio_res["report_month"] = portfolio_res["report_dt"].apply(
        lambda x: str(x)[:7]
    )
    portfolio_res = portfolio_res.merge(FOR, on="report_month", how="left")

    portfolio_res["SSV"] = 0.48

    # Найдем взвешанные ставки по депозитам

    portfolio_res["balance_x_weight_rate"] = (
        portfolio_res["balance"] * portfolio_res["weight_rate"]
    )

    # Вычтем FOR и SSV и перейдем к расчетам

    portfolio_res["ftp"] = (
        portfolio_res["ftp"] * (1 - portfolio_res["FOR"] / 100)
        - portfolio_res["SSV"]
        - portfolio_res["spreads_mat"]
    )

    # Расчет ЧПД

    portfolio_res["margin_year"] = portfolio_res["ftp"] - portfolio_res["weight_rate"]
    portfolio_res["margin_month"] = (1 + (portfolio_res["margin_year"] / 100)) ** (
        1 / 12
    ) - 1
    portfolio_res["margin_day"] = (portfolio_res["margin_year"] / 100) * (1 / 365)
    portfolio_res["report_dt_days"] = portfolio_res["report_dt"].apply(
        lambda x: (x.day)
    )
    portfolio_res["margin_value"] = (
        portfolio_res["balance"]
        * portfolio_res["margin_day"]
        * portfolio_res["report_dt_days"]
    )

    return portfolio_res
