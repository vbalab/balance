import re
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl
from tqdm.notebook import tqdm

from core.models.utils import run_spark_session
from dateutil.relativedelta import relativedelta  # type: ignore[import]
from statsmodels.tsa.seasonal import seasonal_decompose


start_date = "2020-01-01"
end_date = ...  # TODO: use current date - datetime.now()

spark = run_spark_session()


sql = spark.table("prod_dadm_alm_sbx.almde_fl_funds_agg").filter(
    f"report_dt BETWEEN '{start_date}' AND '{end_date}'"
)
current_accounts = pl.DataFrame(sql.toPandas())

sql = spark.table("prod_dadm_alm_sbx.almde_fl_dpst_ftp_rates_vtb").filter(
    f"report_dt BETWEEN '{start_date}' AND '{end_date}'"
)
ftp_rates = pl.DataFrame(sql.toPandas())

key_rate = pl.read_csv("data/key_rate.csv")

# ---- #

group_cols: list[str] = ["salary_flg", "pensioner_flg", "is_vip_or_prv"]

current_accounts = current_accounts.filter(
    pl.col("product_name") == "CURRENT_ACCOUNTS"
).filter(pl.col("currency_iso_cd") == "RUB")

current_accounts = current_accounts.select(
    [
        pl.col("report_dt"),
        pl.col("salary_flg").cast(pl.Int8),
        pl.col("pensioner_flg").cast(pl.Int8),
        pl.col("is_vip_or_prv").cast(pl.Int8),
        pl.col("balance_amt").cast(pl.Float64),
        pl.col("cnt").cast(pl.Int64),
    ]
)

current_accounts = current_accounts.group_by(["report_dt", *group_cols]).agg(
    pl.col("balance_amt").sum(),
    pl.col("cnt").sum(),
)

current_accounts = current_accounts.sort(["report_dt", *group_cols])

salary_map = {"0": "NS", "1": "S"}
pensioner_map = {"0": "NP", "1": "P"}
is_vip_or_prv_map = {"0": "mass", "1": "priv", "2": "vip"}

current_accounts = current_accounts.with_columns(
    pl.col("salary_flg").cast(pl.String).replace(salary_map),
    pl.col("pensioner_flg").cast(pl.String).replace(pensioner_map),
    pl.col("is_vip_or_prv").cast(pl.String).replace(is_vip_or_prv_map),
)

# ---- #

ftp_rates = ftp_rates.with_columns(
    pl.col("vtb_[90d]_ftp_rate").alias("VTB_[90d]_ftp_rate"),
    pl.col("vtb_[365d]_ftp_rate").alias("VTB_[365d]_ftp_rate"),
)

ftp_rates = (
    ftp_rates.sort("report_dt")
    .select(
        "report_dt",
        "VTB_[90d]_ftp_rate",
        "VTB_[365d]_ftp_rate",
    )
    .with_columns(pl.col("report_dt").cast(pl.Date))
)

ftp_rates = ftp_rates.select(
    [
        pl.col(c).alias(
            c.replace('"', "")
            .replace("{", "")
            .replace("}", "")
            .replace("[", "")
            .replace("]", "")
            .replace(",", "_")
        )
        for c in ftp_rates.columns
    ]
)

ftp_rates = (
    ftp_rates.with_columns(
        pl.col("report_dt").dt.month_end(),
    )
    .group_by("report_dt")
    .agg(
        pl.col("VTB_90d_ftp_rate").mean(),
        pl.col("VTB_365d_ftp_rate").mean(),
    )
)

ftp_rates = ftp_rates.sort("report_dt")

# ---- #
key_rate = key_rate.with_columns(pl.col("report_dt").cast(pl.Date))

market_rates = key_rate
market_rates = market_rates.sort("report_dt")


# ---- #
def add_ftp_rates(df_balance: pl.DataFrame, df_ftp: pl.DataFrame) -> pl.DataFrame:
    df_balance = df_balance.clone()

    df_balance = df_balance.join(df_ftp, on="report_dt", how="left")
    # df_balance = df_balance.drop_nulls()
    df_balance = df_balance.sort(
        [
            "report_dt",
            *group_cols,
        ]
    )

    return df_balance


def add_market_rates(
    df_balance: pl.DataFrame, df_market_rates: pl.DataFrame
) -> pl.DataFrame:
    df_balance = df_balance.clone()

    df_balance = df_balance.join(df_market_rates, on="report_dt", how="left")
    # df_balance = df_balance.drop_nulls()
    df_balance = df_balance.sort(
        [
            "report_dt",
            *group_cols,
        ]
    )

    return df_balance


def add_calendar_dummies(df: pl.DataFrame, date_col: str) -> pl.DataFrame:
    def _add_between_dates_dummies(
        name: str, start: date, end: date, ds_col: str
    ) -> pl.Expr:
        return (
            pl.when((pl.col(ds_col) >= pl.lit(start)) & (pl.col(ds_col) <= pl.lit(end)))
            .then(1)
            .otherwise(0)
            .alias(name)
        )

    df = df.with_columns(
        [
            pl.col(date_col).dt.month().cast(pl.Int8).alias("month"),
        ]
    )

    df = df.with_columns(
        [
            _add_between_dates_dummies(
                "covid_20200301_20200531", date(2020, 3, 1), date(2020, 5, 31), date_col
            ),
            _add_between_dates_dummies(
                "svo_20220201_20220331", date(2022, 2, 1), date(2022, 3, 31), date_col
            ),
            _add_between_dates_dummies(
                "svo_20220401_20220731", date(2022, 4, 1), date(2022, 7, 31), date_col
            ),
        ]
    )

    return df


def seasonal_decompose_by_group(
    df: pl.DataFrame,
    y_col: str,
    date_col: str,
    group_cols: list[str],
    period: int = 12,
    model: str = "additive",
):
    """
    Adds a 'seasonal' column to df by running statsmodels.seasonal_decompose
    separately per group. Assumes daily data; uses the first value if multiple
    rows share the same date within a group.
    """
    df = df.to_pandas()

    df["month"] = df["report_dt"].dt.month.astype(str)

    nonadf = df.dropna(subset=[y_col])

    out_frames = []
    for keys, g in nonadf.groupby(group_cols, sort=False, dropna=False):
        s = (
            g.sort_values(date_col).set_index(date_col)[y_col].asfreq("ME")
        )  # monthly freq

        res = seasonal_decompose(
            s, model=model, period=period, extrapolate_trend="freq"
        )

        seasonal_df = res.seasonal.rename("seasonal").to_frame().reset_index()
        g_ret = g.merge(seasonal_df, on=date_col, how="left")

        out_frames.append(g_ret)

    seasonal = pd.concat(out_frames, axis=0, ignore_index=True)
    seasonal["month"] = seasonal["report_dt"].dt.month.astype(str)

    merge_on = ["month", *group_cols]
    seasonal = seasonal[[*merge_on, "seasonal"]]
    seasonal = seasonal.drop_duplicates(merge_on)

    df = df.merge(seasonal, on=merge_on, how="left")
    df = pl.DataFrame(df)
    df = df.drop("month")
    df = df.with_columns(pl.col("report_dt").dt.date())

    return df


rate_offs = [*range(1, 12 + 1)]


def add_ftp_rates_diffs(df: pl.DataFrame, group_cols: Sequence[str]) -> pl.DataFrame:
    df = df.with_columns(
        [
            pl.col("VTB_90d_ftp_rate")
            .diff(n=off)
            .over(group_cols)
            .alias(f"VTB_90_ftp_rate_diff{off}")
            for off in rate_offs
        ]
    )
    df = df.with_columns(
        [
            pl.col("VTB_365d_ftp_rate")
            .diff(n=off)
            .over(group_cols)
            .alias(f"VTB_365_ftp_rate_diff{off}")
            for off in rate_offs
        ]
    )

    return df


def add_market_rates_diffs(df: pl.DataFrame, group_cols: Sequence[str]) -> pl.DataFrame:
    df = df.with_columns(
        [
            pl.col("key_rate").diff(n=off).over(group_cols).alias(f"key_rate_diff{off}")
            for off in rate_offs
        ]
    )

    return df


def create_static_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.clone()
    df = df.sort(["report_dt", *group_cols])

    df = add_calendar_dummies(df, "report_dt")
    df = seasonal_decompose_by_group(
        df, "diff_log_balance_amt", "report_dt", group_cols
    )
    df = add_ftp_rates_diffs(df, group_cols)
    df = add_market_rates_diffs(df, group_cols)

    return df


def add_diff_log(
    df: pl.DataFrame, target_col: str, group_cols: Sequence[str]
) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col(target_col)).log().alias(f"log_{target_col}"),
    )

    df = df.with_columns(
        pl.col(f"log_{target_col}")
        .diff()
        .over(group_cols)
        .alias(f"diff_log_{target_col}"),
    )

    return df


def add_lags(
    df: pl.DataFrame,
    y_col: str,
    date_col: str,
    group_cols: Sequence[str],
    offsets: Sequence[str],
) -> pl.DataFrame:
    def _normalize_offset(off: str) -> str:
        s = off.strip().lower()

        m = re.fullmatch(r"(\d+)(y|m|mo|d)", s)
        if not m:
            raise ValueError(f"Unsupported offset string: {off!r}")

        num, unit = m.groups()
        unit = "mo" if unit == "m" else unit
        return f"{int(num)}{unit}"

    def _make_lag_col_name(off: str) -> str:
        return f"{y_col}_lag_{off.replace(' ', '')}"

    df_cast = df.with_columns(pl.col(date_col).cast(pl.Date).alias(date_col))
    sort_keys = list(group_cols) + [date_col]
    df_sorted = df_cast.sort(sort_keys)
    right = df_sorted.select([*group_cols, date_col, y_col])
    out = df_sorted

    for off in offsets:
        norm = _normalize_offset(off)
        lag_key = f"__lag_key__{off.replace(' ', '')}"
        left = out.with_columns(
            pl.col(date_col).dt.offset_by(f"-{norm}").alias(lag_key)
        ).sort(sort_keys)
        left_cols = left.columns
        joined = left.join_asof(
            right.sort(sort_keys),
            left_on=lag_key,
            right_on=date_col,
            by=group_cols if group_cols else None,
            strategy="backward",
            suffix="__r",
        )
        lag_col = _make_lag_col_name(off)
        joined = (
            joined.rename({f"{y_col}__r": lag_col})
            .select([*left_cols, lag_col])
            .drop(lag_key)
        )
        out = joined

    return out


def add_cumulative_lags(
    df: pl.DataFrame,
    y_col: str,
    date_col: str,
    group_cols: Sequence[str],
    offsets: Sequence[int],
) -> pl.DataFrame:
    df = df.sort([date_col, *group_cols])

    df = df.with_columns(
        [
            pl.col(y_col)
            .diff(n=off)
            .shift(1)
            .over(group_cols)
            .alias(f"{y_col}_diff{off}_lag_1m")
            for off in offsets
        ]
    )

    return df


target_lags = [f"{i}m" for i in range(1, 12 + 1)]
cum_target_lags = [3, 6, 12]

n_accounts_lags = [f"{i}m" for i in range(1, 12 + 1)]
cum_n_accounts_lags = [3, 6, 12]


def create_target_based_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.clone()
    df = df.sort(["report_dt", *group_cols])

    df = add_diff_log(df, "balance_amt", group_cols)
    df = add_lags(df, "diff_log_balance_amt", "report_dt", group_cols, target_lags)
    df = add_cumulative_lags(
        df, "log_balance_amt", "report_dt", group_cols, cum_target_lags
    )

    # ---

    df = add_diff_log(df, "n_accounts", group_cols)
    df = add_lags(df, "diff_log_n_accounts", "report_dt", group_cols, n_accounts_lags)
    df = add_cumulative_lags(
        df, "log_n_accounts", "report_dt", group_cols, cum_n_accounts_lags
    )
    df = df.with_columns(
        [
            pl.col("log_n_accounts")
            .shift(1)
            .over(group_cols)
            .alias("log_n_accounts_lag_1m")
        ]
    )

    return df


# ---- #
TARGET_COL = "diff_log_balance_amt"

NUMERICAL_FEATURES = (
    [
        "seasonal",
        "covid_20200301_20200531",
        "svo_20220201_20220331",
        "svo_20220401_20220731",
    ]
    + [f"VTB_90_ftp_rate_diff{i}" for i in rate_offs]
    + [f"VTB_365_ftp_rate_diff{i}" for i in rate_offs]
    + [f"key_rate_diff{i}" for i in rate_offs]
)

CATEGORICAL_FEATURES = [
    "salary_flg",
    "pensioner_flg",
    "is_vip_or_prv",
]

TARGET_LAG_FEATURES = [f"diff_log_balance_amt_lag_{i}" for i in target_lags] + [
    f"log_balance_amt_diff{i}_lag_1m" for i in cum_target_lags
]
N_ACCOUNTS_LAG_FEATURES = (
    [f"diff_log_n_accounts_lag_{i}" for i in n_accounts_lags]
    + [f"log_n_accounts_diff{i}_lag_1m" for i in cum_n_accounts_lags]
    + ["log_n_accounts_lag_1m"]
)
LAG_FEATURES = TARGET_LAG_FEATURES + N_ACCOUNTS_LAG_FEATURES

PCA_FEATURES = NUMERICAL_FEATURES + LAG_FEATURES
NON_PCA_FEATURES = CATEGORICAL_FEATURES


def TimeTrainTestSplitDate(
    df: pl.DataFrame, split_date: date
) -> tuple[pl.DataFrame, pl.DataFrame]:
    df = df.clone()

    df_train = df.filter(pl.col("report_dt") < split_date)
    df_test = df.filter(pl.col("report_dt") >= split_date)

    return df_train, df_test


def make_test_rows(start: date, end: date) -> pl.DataFrame:
    dates = pl.DataFrame(
        {"report_dt": pl.date_range(start, end, "1mo", eager=True).dt.month_end()}
    )

    salary = pl.DataFrame(
        {"salary_flg": pl.Series(salary_map.values(), dtype=pl.String)}
    )
    pension = pl.DataFrame(
        {"pensioner_flg": pl.Series(pensioner_map.values(), dtype=pl.String)}
    )
    vip = pl.DataFrame(
        {"is_vip_or_prv": pl.Series(is_vip_or_prv_map.values(), dtype=pl.String)}
    )

    combos = salary.join(pension, how="cross").join(vip, how="cross")
    return dates.join(combos, how="cross").sort(
        ["report_dt", "salary_flg", "pensioner_flg", "is_vip_or_prv"]
    )


def concat_new_test_batch(
    df_train: pl.DataFrame, df_test: pl.DataFrame
) -> pl.DataFrame:
    df_test_aligned = df_test.select(
        [
            pl.col(c) if c in df_test.columns else pl.lit(None).alias(c)
            for c in df_train.columns
        ]
    )

    df = pl.concat([df_train, df_test_aligned], how="vertical")

    return df


def duplicate_from_date_forward(
    df: pl.DataFrame,
    date_col: str,
    start_from: str | pl.Date,
    cols_to_copy: Sequence[str],
    group_cols: Sequence[str],
) -> pl.DataFrame:
    df = df.with_columns(pl.col(date_col).cast(pl.Date))

    start_from = (
        df.filter(pl.col(date_col) >= start_from).select(pl.col(date_col).min()).item()
    )
    select_cols = [*group_cols, *cols_to_copy]
    base_vals = df.filter(pl.col(date_col) == pl.lit(start_from)).select(select_cols)

    df_out = (
        df.join(base_vals, on=group_cols, how="left")
        .with_columns(
            [
                pl.when(pl.col(date_col) >= pl.lit(start_from))
                .then(pl.col(f"{c}_right"))
                .otherwise(pl.col(c))
                .alias(c)
                for c in cols_to_copy
            ]
        )
        .select(df.columns)
    )

    return df_out


def create_sample(
    current_accounts_train: pl.DataFrame,
    ftp_rates_train: pl.DataFrame,
    ftp_rates_test: pl.DataFrame,
    market_rates_train: pl.DataFrame,
    market_rates_test: pl.DataFrame,
    start_date: date,
    end_date: date,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    df_train = add_ftp_rates(current_accounts_train, ftp_rates_train)
    df_train = add_market_rates(df_train, market_rates_train)

    df_test = make_test_rows(start_date, end_date - relativedelta(days=1))
    df_test = add_ftp_rates(df_test, ftp_rates_test)
    df_test = add_market_rates(df_test, market_rates_test)

    df = concat_new_test_batch(df_train, df_test)

    df = create_target_based_features(df)
    df = create_static_features(df)

    df = duplicate_from_date_forward(
        df, "report_dt", start_date, LAG_FEATURES, group_cols
    )
    df = df.sort(["report_dt", *group_cols])

    df_train, df_test = TimeTrainTestSplitDate(df, start_date)

    return df_train, df_test


...

# ---- #
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Lasso


def GetPipeline() -> Pipeline:
    pca_numerical = Pipeline(
        [
            ("scaler", StandardScaler()),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("num_part", pca_numerical, PCA_FEATURES),
            ("cat_part", OneHotEncoder(drop="first"), NON_PCA_FEATURES),
        ]
    )

    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("linreg", Lasso(alpha=0.04)),
        ]
    )


def train_linreg(df_train: pl.DataFrame, n_months: int) -> list[Pipeline]:
    models: list[Pipeline] = []

    for horizon in range(n_months):
        df_tmp = df_train.with_columns(
            [pl.col(c).shift(horizon).over(group_cols) for c in LAG_FEATURES]
        )
        df_tmp = df_tmp.drop_nulls()
        df_tmp = df_tmp.to_pandas()

        X, y = df_tmp[PCA_FEATURES + group_cols], df_tmp[TARGET_COL]

        model = GetPipeline()
        model.fit(X, y)
        models.append(model)

    return models


def test_linreg(
    df_test: pl.DataFrame, models: Sequence[Pipeline], n_months: int
) -> list[float]:
    y_pred: list[float] = []

    dates = (
        df_test.select(pl.col("report_dt").unique().sort())
        .cast(pl.String)
        .to_series()
        .to_list()
    )
    assert len(dates) == n_months

    X_test = (
        df_test.to_pandas()
        .set_index("report_dt")
        .loc[:, NUMERICAL_FEATURES + LAG_FEATURES + group_cols]
    )

    for horizon in range(n_months):
        X_test_horizon = X_test.loc[X_test.index == dates[horizon]]

        y_pred += models[horizon].predict(X_test_horizon).tolist()

    return y_pred


def build_level_pred(
    df: pd.DataFrame,
    log_target_col: str,
    diff_log_target_pred_col: str,
    group_cols: list[str],
    time_col: str,
) -> pd.DataFrame:
    """
    Reconstruct level predictions from diff(log(target)) predictions per group.
    Requires columns: 'log_target', 'diff_log_target_pred'.
    Creates: 'log_target_pred', 'target_pred'.
    """
    log_target_pred_col = diff_log_target_pred_col[5:]
    target_pred_col = log_target_pred_col[4:]
    df = df.sort_values(group_cols + [time_col]).copy()
    df[log_target_pred_col] = np.nan

    def _per_group(g: pd.DataFrame) -> pd.DataFrame:
        s = g[diff_log_target_pred_col]

        first_idx = s.index[s.notna()][0]
        pos = g.index.get_loc(first_idx)

        # base = last actual log_target before prediction start
        base = (
            g[log_target_col].iloc[pos - 1] if pos > 0 else g[log_target_col].iloc[pos]
        )

        # cumulative sum of diffs from the first prediction onwards
        csum = s.iloc[pos:].cumsum()
        g.loc[csum.index, log_target_pred_col] = base + csum

        return g

    df = df.groupby(group_cols, group_keys=False).apply(_per_group)
    df[target_pred_col] = np.exp(df[log_target_pred_col])

    return df


def build_test_prediction_model(
    df_train: pl.DataFrame, df_test: pl.DataFrame, y_pred: Sequence[float]
) -> pd.DataFrame:
    df = pl.concat([df_train, df_test], how="vertical")
    df = df.to_pandas()

    df_test = df_test.to_pandas()

    df_test["diff_log_balance_amt_pred"] = pd.Series(y_pred)
    df_test = df_test[["report_dt", *group_cols, "diff_log_balance_amt_pred"]]

    df = df.merge(df_test, on=["report_dt", *group_cols], how="left")
    df = build_level_pred(
        df, "log_balance_amt", "diff_log_balance_amt_pred", group_cols, "report_dt"
    )
    df = df[["report_dt", *group_cols, "balance_amt_pred"]].dropna()

    return df


# ---- #

# TODO: ftp_rates_scenario_path & market_rates_scenario_path should be added as args when doing python modelling.py with default arguments of paths
ftp_rates_scenario = ...
market_rates_scenario = ...


start_test_date = date(2024, 8, 1)
T = 1
n_months = 12

linreg_preds = []

for t in tqdm(range(T)):
    test_batch_start = start_test_date + relativedelta(months=t)
    test_batch_end = test_batch_start + relativedelta(months=n_months)

    current_accounts_train, _ = TimeTrainTestSplitDate(
        current_accounts, test_batch_start
    )

    df_train, df_test = create_sample(
        current_accounts_train,
        ftp_rates,
        ftp_rates_scenario,
        market_rates,
        market_rates_scenario,
        test_batch_start,
        test_batch_end,
    )

    models = train_linreg(df_train, n_months)
    y_pred = test_linreg(df_test, models, n_months)

    df = build_test_prediction_model(df_train, df_test, y_pred)

    linreg_preds.append(df.copy())

linreg_prediction = pd.concat(linreg_preds, axis=0)
