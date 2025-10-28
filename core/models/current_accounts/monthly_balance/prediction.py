from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import os
from pickle import dump
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl
import pyspark.sql.functions as F
from dateutil.relativedelta import relativedelta
from pandas import DataFrame
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

from core.models.utils import convert_decimals, dt_convert, check_existence
from core.upfm.commons import (
    BaseModel,
    DataLoader,
    ForecastContext,
    ModelInfo,
    ModelMetaInfo,
    ModelTrainer,
    _REPORT_DT_COLUMN,
)


GROUP_COLS: List[str] = ["salary_flg", "pensioner_flg", "is_vip_or_prv"]

SALARY_MAP: Dict[str, str] = {"0": "NS", "1": "S"}
PENSIONER_MAP: Dict[str, str] = {"0": "NP", "1": "P"}
VIP_MAP: Dict[str, str] = {"0": "mass", "1": "priv", "2": "vip"}

RATE_OFFSETS: List[int] = list(range(1, 12 + 1))
TARGET_LAGS: List[str] = [f"{i}m" for i in range(1, 12 + 1)]
CUM_TARGET_LAGS: List[int] = [3, 6, 12]
N_ACCOUNTS_LAGS: List[str] = [f"{i}m" for i in range(1, 12 + 1)]
CUM_N_ACCOUNTS_LAGS: List[int] = [3, 6, 12]

TARGET_COL = "diff_log_balance_amt"

NUMERICAL_FEATURES: List[str] = (
    [
        "seasonal",
        "covid_20200301_20200531",
        "svo_20220201_20220331",
        "svo_20220401_20220731",
    ]
    + [f"VTB_90_ftp_rate_diff{i}" for i in RATE_OFFSETS]
    + [f"VTB_365_ftp_rate_diff{i}" for i in RATE_OFFSETS]
    + [f"key_rate_diff{i}" for i in RATE_OFFSETS]
)

CATEGORICAL_FEATURES: List[str] = GROUP_COLS.copy()

TARGET_LAG_FEATURES: List[str] = [
    f"diff_log_balance_amt_lag_{i}" for i in TARGET_LAGS
] + [f"log_balance_amt_diff{i}_lag_1m" for i in CUM_TARGET_LAGS]

N_ACCOUNTS_LAG_FEATURES: List[str] = [
    f"diff_log_n_accounts_lag_{i}" for i in N_ACCOUNTS_LAGS
] + [f"log_n_accounts_diff{i}_lag_1m" for i in CUM_N_ACCOUNTS_LAGS] + [
    "log_n_accounts_lag_1m"
]

LAG_FEATURES: List[str] = TARGET_LAG_FEATURES + N_ACCOUNTS_LAG_FEATURES

PCA_FEATURES: List[str] = NUMERICAL_FEATURES + LAG_FEATURES
NON_PCA_FEATURES: List[str] = CATEGORICAL_FEATURES


def _ensure_date_series(df: pl.DataFrame, column: str) -> pl.DataFrame:
    return df.with_columns(pl.col(column).cast(pl.Date))


def _filter_by_date_range(
    df: pl.DataFrame, start_date: Optional[datetime | date], end_date: Optional[datetime | date]
) -> pl.DataFrame:
    exprs: List[pl.Expr] = []
    if start_date is not None:
        exprs.append(pl.col("report_dt") >= pl.lit(pd.Timestamp(start_date).date()))
    if end_date is not None:
        exprs.append(pl.col("report_dt") <= pl.lit(pd.Timestamp(end_date).date()))
    if not exprs:
        return df
    mask = exprs[0]
    for expr in exprs[1:]:
        mask = mask & expr
    return df.filter(mask)


def _prepare_current_accounts(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter(pl.col("product_name") == "CURRENT_ACCOUNTS")
    df = df.filter(pl.col("currency_iso_cd") == "RUB")

    df = df.select(
        [
            pl.col("report_dt"),
            pl.col("salary_flg").cast(pl.Int8),
            pl.col("pensioner_flg").cast(pl.Int8),
            pl.col("is_vip_or_prv").cast(pl.Int8),
            pl.col("balance_amt").cast(pl.Float64),
            pl.col("cnt").cast(pl.Float64).alias("n_accounts"),
        ]
    )

    df = (
        df.groupby(["report_dt", *GROUP_COLS])
        .agg(pl.col("balance_amt").sum(), pl.col("n_accounts").sum())
        .sort(["report_dt", *GROUP_COLS])
    )

    df = df.with_columns(
        pl.col("salary_flg").cast(pl.String).replace(SALARY_MAP),
        pl.col("pensioner_flg").cast(pl.String).replace(PENSIONER_MAP),
        pl.col("is_vip_or_prv").cast(pl.String).replace(VIP_MAP),
    )

    return _ensure_date_series(df, "report_dt")


def _sanitize_ftp_columns(columns: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for column in columns:
        new_name = (
            column.replace('"', "")
            .replace("{", "")
            .replace("}", "")
            .replace("[", "")
            .replace("]", "")
            .replace(",", "_")
        )
        cleaned.append(new_name)
    return cleaned


def _prepare_ftp_rates(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort("report_dt")
    df = df.select(
        "report_dt",
        pl.col("vtb_[90d]_ftp_rate").alias("VTB_[90d]_ftp_rate"),
        pl.col("vtb_[365d]_ftp_rate").alias("VTB_[365d]_ftp_rate"),
    )
    df = _ensure_date_series(df, "report_dt")
    df = df.rename({c: n for c, n in zip(df.columns, _sanitize_ftp_columns(df.columns))})
    df = (
        df.with_columns(pl.col("report_dt").dt.month_end())
        .groupby("report_dt")
        .agg(
            pl.col("VTB_90d_ftp_rate").mean(),
            pl.col("VTB_365d_ftp_rate").mean(),
        )
        .sort("report_dt")
    )
    return df


def _prepare_market_rates(df: pl.DataFrame) -> pl.DataFrame:
    df = _ensure_date_series(df, "report_dt")
    df = df.sort("report_dt")
    return df


def add_ftp_rates(df_balance: pl.DataFrame, df_ftp: pl.DataFrame) -> pl.DataFrame:
    return df_balance.join(df_ftp, on="report_dt", how="left").sort(["report_dt", *GROUP_COLS])


def add_market_rates(df_balance: pl.DataFrame, df_market: pl.DataFrame) -> pl.DataFrame:
    return df_balance.join(df_market, on="report_dt", how="left").sort(["report_dt", *GROUP_COLS])


def add_calendar_dummies(df: pl.DataFrame, date_col: str) -> pl.DataFrame:
    def _add_between(name: str, start: date, end: date, ds_col: str) -> pl.Expr:
        return (
            pl.when((pl.col(ds_col) >= pl.lit(start)) & (pl.col(ds_col) <= pl.lit(end)))
            .then(1)
            .otherwise(0)
            .alias(name)
        )

    df = df.with_columns(pl.col(date_col).dt.month().cast(pl.Int8).alias("month"))
    df = df.with_columns(
        _add_between("covid_20200301_20200531", date(2020, 3, 1), date(2020, 5, 31), date_col),
        _add_between("svo_20220201_20220331", date(2022, 2, 1), date(2022, 3, 31), date_col),
        _add_between("svo_20220401_20220731", date(2022, 4, 1), date(2022, 7, 31), date_col),
    )
    return df


def seasonal_decompose_by_group(
    df: pl.DataFrame,
    y_col: str,
    date_col: str,
    group_cols: Sequence[str],
    period: int = 12,
    model: str = "additive",
) -> pl.DataFrame:
    df_pd = df.to_pandas()
    df_pd["month"] = df_pd[date_col].dt.month.astype(str)
    non_null = df_pd.dropna(subset=[y_col])

    out_frames: List[pd.DataFrame] = []
    for _, group_df in non_null.groupby(list(group_cols), sort=False, dropna=False):
        series = group_df.sort_values(date_col).set_index(date_col)[y_col].asfreq("ME")
        res = seasonal_decompose(series, model=model, period=period, extrapolate_trend="freq")
        seasonal_df = res.seasonal.rename("seasonal").to_frame().reset_index()
        out_frames.append(group_df.merge(seasonal_df, on=date_col, how="left"))

    seasonal = pd.concat(out_frames, axis=0, ignore_index=True)
    seasonal["month"] = seasonal[date_col].dt.month.astype(str)
    merge_on = ["month", *group_cols]
    seasonal = seasonal[[*merge_on, "seasonal"]].drop_duplicates(merge_on)

    merged = df_pd.merge(seasonal, on=merge_on, how="left")
    merged = merged.drop(columns="month")
    merged[date_col] = merged[date_col].dt.date
    return pl.DataFrame(merged)


def add_ftp_rates_diffs(df: pl.DataFrame, group_cols: Sequence[str]) -> pl.DataFrame:
    df = df.with_columns(
        [
            pl.col("VTB_90d_ftp_rate").diff(n=offset).over(group_cols).alias(f"VTB_90_ftp_rate_diff{offset}")
            for offset in RATE_OFFSETS
        ]
    )
    df = df.with_columns(
        [
            pl.col("VTB_365d_ftp_rate").diff(n=offset).over(group_cols).alias(
                f"VTB_365_ftp_rate_diff{offset}"
            )
            for offset in RATE_OFFSETS
        ]
    )
    return df


def add_market_rates_diffs(df: pl.DataFrame, group_cols: Sequence[str]) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col("key_rate").diff(n=offset).over(group_cols).alias(f"key_rate_diff{offset}")
            for offset in RATE_OFFSETS
        ]
    )


def add_diff_log(df: pl.DataFrame, target_col: str, group_cols: Sequence[str]) -> pl.DataFrame:
    safe_col = pl.when(pl.col(target_col) > 0).then(pl.col(target_col)).otherwise(None)
    df = df.with_columns(safe_col.log().alias(f"log_{target_col}"))
    df = df.with_columns(
        pl.col(f"log_{target_col}").diff().over(group_cols).alias(f"diff_log_{target_col}")
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
        off_str = off.strip().lower()
        match = re.fullmatch(r"(\d+)(y|m|mo|d)", off_str)
        if match is None:
            raise ValueError(f"Unsupported offset string: {off!r}")
        num, unit = match.groups()
        unit = "mo" if unit == "m" else unit
        return f"{int(num)}{unit}"

    def _make_lag_name(off: str) -> str:
        return f"{y_col}_lag_{off.replace(' ', '')}"

    df_cast = df.with_columns(pl.col(date_col).cast(pl.Date).alias(date_col))
    sort_keys = [*group_cols, date_col]
    df_sorted = df_cast.sort(sort_keys)
    right = df_sorted.select([*group_cols, date_col, y_col])
    out = df_sorted

    for off in offsets:
        normalized = _normalize_offset(off)
        lag_key = f"__lag_key__{off.replace(' ', '')}"
        left = out.with_columns(pl.col(date_col).dt.offset_by(f"-{normalized}").alias(lag_key)).sort(
            sort_keys
        )
        left_columns = left.columns
        joined = left.join_asof(
            right.sort(sort_keys),
            left_on=lag_key,
            right_on=date_col,
            by=group_cols if group_cols else None,
            strategy="backward",
            suffix="__r",
        )
        lag_col = _make_lag_name(off)
        joined = joined.rename({f"{y_col}__r": lag_col}).select([*left_columns, lag_col]).drop(lag_key)
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
    return df.with_columns(
        [
            pl.col(y_col)
            .diff(n=offset)
            .shift(1)
            .over(group_cols)
            .alias(f"{y_col}_diff{offset}_lag_1m")
            for offset in offsets
        ]
    )


def create_target_based_features(df: pl.DataFrame) -> pl.DataFrame:
    df = add_diff_log(df, "balance_amt", GROUP_COLS)
    df = add_lags(df, "diff_log_balance_amt", "report_dt", GROUP_COLS, TARGET_LAGS)
    df = add_cumulative_lags(df, "log_balance_amt", "report_dt", GROUP_COLS, CUM_TARGET_LAGS)

    df = add_diff_log(df, "n_accounts", GROUP_COLS)
    df = add_lags(df, "diff_log_n_accounts", "report_dt", GROUP_COLS, N_ACCOUNTS_LAGS)
    df = add_cumulative_lags(df, "log_n_accounts", "report_dt", GROUP_COLS, CUM_N_ACCOUNTS_LAGS)
    df = df.with_columns(
        pl.col("log_n_accounts").shift(1).over(GROUP_COLS).alias("log_n_accounts_lag_1m")
    )
    return df


def create_static_features(df: pl.DataFrame) -> pl.DataFrame:
    df = add_calendar_dummies(df, "report_dt")
    df = seasonal_decompose_by_group(df, "diff_log_balance_amt", "report_dt", GROUP_COLS)
    df = add_ftp_rates_diffs(df, GROUP_COLS)
    df = add_market_rates_diffs(df, GROUP_COLS)
    return df


def time_train_test_split(df: pl.DataFrame, split_date: date) -> Tuple[pl.DataFrame, pl.DataFrame]:
    df_train = df.filter(pl.col("report_dt") < split_date)
    df_test = df.filter(pl.col("report_dt") >= split_date)
    return df_train, df_test


def make_test_rows(start: date, end: date) -> pl.DataFrame:
    dates = pl.DataFrame({"report_dt": pl.date_range(start, end, "1mo", eager=True).dt.month_end()})
    salary = pl.DataFrame({"salary_flg": pl.Series(SALARY_MAP.values(), dtype=pl.String)})
    pension = pl.DataFrame({"pensioner_flg": pl.Series(PENSIONER_MAP.values(), dtype=pl.String)})
    vip = pl.DataFrame({"is_vip_or_prv": pl.Series(VIP_MAP.values(), dtype=pl.String)})
    combos = salary.join(pension, how="cross").join(vip, how="cross")
    return dates.join(combos, how="cross").sort(["report_dt", *GROUP_COLS])


def concat_new_test_batch(df_train: pl.DataFrame, df_test: pl.DataFrame) -> pl.DataFrame:
    df_test_aligned = df_test.select(
        [pl.col(c) if c in df_test.columns else pl.lit(None).alias(c) for c in df_train.columns]
    )
    return pl.concat([df_train, df_test_aligned], how="vertical")


def duplicate_from_date_forward(
    df: pl.DataFrame,
    date_col: str,
    start_from: date,
    cols_to_copy: Sequence[str],
    group_cols: Sequence[str],
) -> pl.DataFrame:
    df = df.with_columns(pl.col(date_col).cast(pl.Date))
    start_value = (
        df.filter(pl.col(date_col) >= pl.lit(start_from)).select(pl.col(date_col).min()).item()
    )
    select_cols = [*group_cols, *cols_to_copy]
    base_vals = df.filter(pl.col(date_col) == pl.lit(start_value)).select(select_cols)
    df_out = (
        df.join(base_vals, on=group_cols, how="left")
        .with_columns(
            [
                pl.when(pl.col(date_col) >= pl.lit(start_value))
                .then(pl.col(f"{col}_right"))
                .otherwise(pl.col(col))
                .alias(col)
                for col in cols_to_copy
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
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    df_train = add_ftp_rates(current_accounts_train, ftp_rates_train)
    df_train = add_market_rates(df_train, market_rates_train)

    df_test = make_test_rows(start_date, end_date - relativedelta(days=1))
    df_test = add_ftp_rates(df_test, ftp_rates_test)
    df_test = add_market_rates(df_test, market_rates_test)

    df = concat_new_test_batch(df_train, df_test)
    df = create_target_based_features(df)
    df = create_static_features(df)
    df = duplicate_from_date_forward(df, "report_dt", start_date, LAG_FEATURES, GROUP_COLS)
    df = df.sort(["report_dt", *GROUP_COLS])

    df_train_final, df_test_final = time_train_test_split(df, start_date)
    return df_train_final, df_test_final


def build_training_dataset(
    current_accounts: pl.DataFrame, ftp_rates: pl.DataFrame, market_rates: pl.DataFrame
) -> pl.DataFrame:
    df = add_ftp_rates(current_accounts, ftp_rates)
    df = add_market_rates(df, market_rates)
    df = create_target_based_features(df)
    df = create_static_features(df)
    return df.sort(["report_dt", *GROUP_COLS])


def build_level_pred(
    df: pd.DataFrame,
    log_target_col: str,
    diff_log_target_pred_col: str,
    group_cols: Sequence[str],
    time_col: str,
) -> pd.DataFrame:
    log_target_pred_col = diff_log_target_pred_col[5:]
    target_pred_col = log_target_pred_col[4:]
    df = df.sort_values([*group_cols, time_col]).copy()
    df[log_target_pred_col] = np.nan

    def _per_group(g: pd.DataFrame) -> pd.DataFrame:
        series = g[diff_log_target_pred_col]
        valid_idx = series.index[series.notna()]
        if len(valid_idx) == 0:
            return g
        first_idx = valid_idx[0]
        pos = g.index.get_loc(first_idx)
        base = g[log_target_col].iloc[pos - 1] if pos > 0 else g[log_target_col].iloc[pos]
        csum = series.iloc[pos:].cumsum()
        g.loc[csum.index, log_target_pred_col] = base + csum
        return g

    df = df.groupby(list(group_cols), group_keys=False).apply(_per_group)
    df[target_pred_col] = np.exp(df[log_target_pred_col])
    return df


def build_test_prediction_model(
    df_train: pl.DataFrame, df_test: pl.DataFrame, y_pred: Sequence[float]
) -> pd.DataFrame:
    df = pl.concat([df_train, df_test], how="vertical").to_pandas()
    df_test_pd = df_test.to_pandas()
    df_test_pd["diff_log_balance_amt_pred"] = pd.Series(y_pred)
    df_test_pd = df_test_pd[["report_dt", *GROUP_COLS, "diff_log_balance_amt_pred"]]
    df = df.merge(df_test_pd, on=["report_dt", *GROUP_COLS], how="left")
    df = build_level_pred(df, "log_balance_amt", "diff_log_balance_amt_pred", GROUP_COLS, "report_dt")
    df = df[["report_dt", *GROUP_COLS, "balance_amt_pred"]].dropna()
    return df


def get_pipeline(alpha: float) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            ("num_part", Pipeline([("scaler", StandardScaler())]), PCA_FEATURES),
            ("cat_part", OneHotEncoder(drop="first"), NON_PCA_FEATURES),
        ]
    )
    return Pipeline([("preprocessing", preprocessing), ("linreg", Lasso(alpha=alpha))])


def train_linreg(df_train: pl.DataFrame, n_months: int, alpha: float) -> List[Pipeline]:
    models: List[Pipeline] = []
    for horizon in range(n_months):
        df_tmp = df_train.with_columns(
            [pl.col(col).shift(horizon).over(GROUP_COLS) for col in LAG_FEATURES]
        ).drop_nulls()
        df_pd = df_tmp.to_pandas()
        if df_pd.empty:
            raise ValueError("Training data is insufficient for the requested horizon")
        X = df_pd[PCA_FEATURES + GROUP_COLS]
        y = df_pd[TARGET_COL]
        model = get_pipeline(alpha)
        model.fit(X, y)
        models.append(model)
    return models


def test_linreg(df_test: pl.DataFrame, models: Sequence[Pipeline], n_months: int) -> List[float]:
    y_pred: List[float] = []
    dates = df_test.select(pl.col("report_dt").unique().sort()).to_series().to_list()
    if len(dates) != n_months:
        raise ValueError("Test set horizon does not match requested number of months")
    df_pd = df_test.to_pandas()
    df_pd["report_dt"] = pd.to_datetime(df_pd["report_dt"])
    df_pd = df_pd.sort_values(["report_dt", *GROUP_COLS])
    feature_cols = NUMERICAL_FEATURES + LAG_FEATURES + GROUP_COLS
    for horizon, model in enumerate(models[:n_months]):
        horizon_date = pd.Timestamp(dates[horizon])
        mask = df_pd["report_dt"] == horizon_date
        X_test_horizon = df_pd.loc[mask, feature_cols]
        if X_test_horizon.empty:
            continue
        preds = model.predict(X_test_horizon)
        y_pred.extend(preds.tolist())
    return y_pred


@dataclass
class MonthlyBalanceConfig:
    model_name: str = "current_accounts_monthly_balance"
    current_accounts_table: str = "prod_dadm_alm_sbx.almde_fl_funds_agg"
    ftp_rates_table: str = "prod_dadm_alm_sbx.almde_fl_dpst_ftp_rates_vtb"
    key_rate_path: str = "data/key_rate.csv"
    default_start_date: datetime = datetime(2015, 1, 31)
    horizon: int = 12
    lasso_alpha: float = 0.04


class MonthlyBalanceDataLoader(DataLoader):
    def __init__(self, config: MonthlyBalanceConfig | None = None) -> None:
        self.config = config or MonthlyBalanceConfig()

    def get_maximum_train_range(self, spark: SparkSession) -> Tuple[datetime, datetime]:
        df = convert_decimals(spark.table(self.config.current_accounts_table))
        stats = df.select(
            F.min("report_dt").alias("min_dt"), F.max("report_dt").alias("max_dt")
        ).collect()[0]
        min_dt = stats["min_dt"]
        max_dt = stats["max_dt"]
        if not isinstance(min_dt, datetime) or not isinstance(max_dt, datetime):
            raise TypeError("report_dt column must be datetime")
        return max(min_dt, self.config.default_start_date), max_dt

    def _load_current_accounts(
        self,
        spark: SparkSession,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> pl.DataFrame:
        df = convert_decimals(spark.table(self.config.current_accounts_table))
        if start_date is not None:
            df = df.filter(F.col("report_dt") >= F.lit(start_date))
        if end_date is not None:
            df = df.filter(F.col("report_dt") <= F.lit(end_date))
        pdf = df.select(
            "report_dt",
            "salary_flg",
            "pensioner_flg",
            "is_vip_or_prv",
            "balance_amt",
            "cnt",
            "product_name",
            "currency_iso_cd",
        ).toPandas()
        pdf["report_dt"] = pd.to_datetime(pdf["report_dt"])
        return _prepare_current_accounts(pl.DataFrame(pdf))

    def _load_ftp_rates(
        self,
        spark: SparkSession,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> pl.DataFrame:
        df = convert_decimals(spark.table(self.config.ftp_rates_table))
        if start_date is not None:
            df = df.filter(F.col("report_dt") >= F.lit(start_date))
        if end_date is not None:
            df = df.filter(F.col("report_dt") <= F.lit(end_date))
        pdf = df.select("report_dt", "vtb_[90d]_ftp_rate", "vtb_[365d]_ftp_rate").toPandas()
        pdf["report_dt"] = pd.to_datetime(pdf["report_dt"])
        return _prepare_ftp_rates(pl.DataFrame(pdf))

    def _load_market_rates(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> pl.DataFrame:
        if not os.path.exists(self.config.key_rate_path):
            raise FileNotFoundError(
                f"Key rate data not found at {self.config.key_rate_path}. Provide updated configuration."
            )
        df = pl.read_csv(self.config.key_rate_path)
        df = df.rename({df.columns[0]: "report_dt"}) if df.columns[0] != "report_dt" else df
        df = _prepare_market_rates(df)
        return _filter_by_date_range(df, start_date, end_date)

    def get_training_data(
        self,
        spark: SparkSession,
        start_date: datetime,
        end_date: datetime,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, DataFrame]:
        current_accounts = self._load_current_accounts(spark, start_date, end_date)
        ftp_rates = self._load_ftp_rates(spark, start_date, end_date)
        market_rates = self._load_market_rates(start_date, end_date)
        return {
            "current_accounts": current_accounts.to_pandas(),
            "ftp_rates": ftp_rates.to_pandas(),
            "market_rates": market_rates.to_pandas(),
        }

    def get_prediction_data(
        self,
        spark: SparkSession,
        start_date: datetime,
        end_date: datetime,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, DataFrame]:
        current_accounts = self._load_current_accounts(spark, None, end_date)
        ftp_rates = self._load_ftp_rates(spark, None, end_date)
        market_rates = self._load_market_rates(None, end_date)
        return {
            "current_accounts": current_accounts.to_pandas(),
            "ftp_rates": ftp_rates.to_pandas(),
            "market_rates": market_rates.to_pandas(),
        }

    def get_ground_truth(
        self,
        spark: SparkSession,
        start_date: datetime,
        end_date: datetime,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, DataFrame]:
        current_accounts = self._load_current_accounts(spark, start_date, end_date)
        pdf = current_accounts.to_pandas()[["report_dt", *GROUP_COLS, "balance_amt"]]
        pdf = pdf.set_index("report_dt")
        pdf.index.name = _REPORT_DT_COLUMN
        return {"target": pdf}


class MonthlyBalanceForecaster:
    def __init__(self, config: MonthlyBalanceConfig | None = None) -> None:
        self.config = config or MonthlyBalanceConfig()
        self.models: List[Pipeline] = []
        self.is_fitted: bool = False

    def fit(
        self,
        current_accounts: pl.DataFrame,
        ftp_rates: pl.DataFrame,
        market_rates: pl.DataFrame,
    ) -> "MonthlyBalanceForecaster":
        training_df = build_training_dataset(current_accounts, ftp_rates, market_rates)
        self.models = train_linreg(training_df, self.config.horizon, self.config.lasso_alpha)
        self.is_fitted = True
        return self

    def predict(
        self,
        current_accounts: pl.DataFrame,
        ftp_rates: pl.DataFrame,
        ftp_rates_scenario: pl.DataFrame,
        market_rates: pl.DataFrame,
        market_rates_scenario: pl.DataFrame,
        forecast_start: date,
        horizon: int,
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        if horizon > len(self.models):
            raise ValueError("Horizon exceeds the number of fitted models")
        df_train = current_accounts.filter(pl.col("report_dt") < pl.lit(forecast_start))
        ftp_train = ftp_rates.filter(pl.col("report_dt") < pl.lit(forecast_start))
        market_train = market_rates.filter(pl.col("report_dt") < pl.lit(forecast_start))

        forecast_end = forecast_start + relativedelta(months=horizon)
        df_train_feat, df_test_feat = create_sample(
            df_train,
            ftp_train,
            ftp_rates_scenario,
            market_train,
            market_rates_scenario,
            forecast_start,
            forecast_end,
        )
        preds = test_linreg(df_test_feat, self.models, horizon)
        return build_test_prediction_model(df_train_feat, df_test_feat, preds)


class MonthlyBalanceModelTrainer(ModelTrainer):
    def __init__(self, config: MonthlyBalanceConfig | None = None) -> None:
        self.config = config or MonthlyBalanceConfig()
        self.dataloader = MonthlyBalanceDataLoader(self.config)

    def get_trained_model(
        self,
        spark: SparkSession,
        end_date: datetime,
        start_date: Optional[datetime] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> MonthlyBalanceForecaster:
        if start_date is None:
            start_date = self.config.default_start_date
        data = self.dataloader.get_training_data(spark, start_date, end_date)
        current_accounts = pl.DataFrame(data["current_accounts"]).with_columns(
            pl.col("report_dt").cast(pl.Date)
        )
        ftp_rates = pl.DataFrame(data["ftp_rates"]).with_columns(pl.col("report_dt").cast(pl.Date))
        market_rates = pl.DataFrame(data["market_rates"]).with_columns(
            pl.col("report_dt").cast(pl.Date)
        )
        model = MonthlyBalanceForecaster(self.config)
        model.fit(current_accounts, ftp_rates, market_rates)
        return model

    def save_trained_model(
        self,
        spark: SparkSession,
        saving_path: str,
        end_date: datetime,
        start_date: Optional[datetime] = None,
        overwrite: bool = True,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        model = self.get_trained_model(spark, end_date, start_date, hyperparams)
        if start_date is None:
            start_date = self.config.default_start_date
        filename = (
            f"{self.config.model_name}_{dt_convert(start_date)}_{dt_convert(end_date)}.pickle"
        )
        if check_existence(saving_path, filename, overwrite=overwrite):
            return filename
        os.makedirs(saving_path, exist_ok=True)
        with open(os.path.join(saving_path, filename), "wb") as file:
            dump(model, file)
        return filename


class MonthlyBalanceModelAdapter(BaseModel):
    def __init__(
        self,
        model_info_: ModelInfo,
        filepath_or_buffer: Any,
        config: MonthlyBalanceConfig | None = None,
    ) -> None:
        self.config = config or MonthlyBalanceConfig()
        super().__init__(model_info_, filepath_or_buffer)

    def predict(
        self,
        forecast_context: ForecastContext,
        portfolio: DataFrame | None = None,
        **params: Any,
    ) -> pd.DataFrame:
        if forecast_context.scenario is None:
            raise ValueError("Scenario is required for prediction")

        model_data = forecast_context.model_data
        required_keys = {"current_accounts", "ftp_rates", "market_rates"}
        if not required_keys.issubset(model_data):
            raise KeyError(
                "Model data must contain current_accounts, ftp_rates and market_rates tables"
            )

        forecast_dates = forecast_context.forecast_dates
        forecast_start = min(forecast_dates)
        horizon = len(forecast_dates)

        scenario_df = forecast_context.scenario.scenario_data.copy()
        scenario_df["report_dt"] = pd.to_datetime(scenario_df["report_dt"])
        forecast_index = pd.DatetimeIndex(forecast_dates)
        scenario_df = scenario_df.set_index("report_dt")
        missing = forecast_index.difference(scenario_df.index)
        if not missing.empty:
            raise KeyError(
                "Scenario data is missing required forecast dates: "
                + ", ".join(missing.strftime("%Y-%m-%d"))
            )
        scenario_df = scenario_df.loc[forecast_index].reset_index()

        ftp_cols = ["report_dt", "VTB_90d_ftp_rate", "VTB_365d_ftp_rate"]
        market_cols = ["report_dt", "key_rate"]
        if not set(ftp_cols).issubset(scenario_df.columns) or not set(market_cols).issubset(
            scenario_df.columns
        ):
            raise KeyError(
                "Scenario data must contain VTB_90d_ftp_rate, VTB_365d_ftp_rate and key_rate columns"
            )

        ftp_scenario = pl.DataFrame(scenario_df[ftp_cols]).with_columns(
            pl.col("report_dt").cast(pl.Date)
        )
        market_scenario = pl.DataFrame(scenario_df[market_cols]).with_columns(
            pl.col("report_dt").cast(pl.Date)
        )

        current_accounts = pl.DataFrame(model_data["current_accounts"]).with_columns(
            pl.col("report_dt").cast(pl.Date)
        )
        ftp_rates = pl.DataFrame(model_data["ftp_rates"]).with_columns(
            pl.col("report_dt").cast(pl.Date)
        )
        market_rates = pl.DataFrame(model_data["market_rates"]).with_columns(
            pl.col("report_dt").cast(pl.Date)
        )

        return self._model_meta.predict(
            current_accounts=current_accounts,
            ftp_rates=ftp_rates,
            ftp_rates_scenario=ftp_scenario,
            market_rates=market_rates,
            market_rates_scenario=market_scenario,
            forecast_start=forecast_start.date(),
            horizon=horizon,
        )


MonthlyBalancePrediction = ModelMetaInfo(
    model_name=MonthlyBalanceConfig().model_name,
    model_trainer=MonthlyBalanceModelTrainer(),
    data_loader=MonthlyBalanceDataLoader(),
    adapter=MonthlyBalanceModelAdapter,
    segment=None,
)
