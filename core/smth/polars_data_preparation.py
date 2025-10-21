import os
import json
import logging
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import fs
from itertools import product
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Union


MORTGAGE_PRODUCT_REGEX_RULES = {
    "state_support": "(?i)гос",
    "PNF": "(?i)пнф",
    "refinance": "(?i)рефин",
    "SFR": "(?i)сфр",
    "under_construction": "(?i)строящ|сж",
    "finished_construction": "(?i)готов|гж",
    "far_east": "(?i)дальневост",
    "children": "(?i)семей|детск",
    "military": "(?i)воен",
    "investment": "(?i)инвест",
    "BFKO": "(?i)БФКО",
    # "village": "(?i)сельск",
}

LOAN_IDENTIFIER = "histor_agreement"


def add_month_year_columns(
    data: pl.DataFrame, date_labels: Sequence[str] = ["report_dt", "open_dt"]
) -> pl.DataFrame:
    """
    Adds separate columns for month and year to the data for each date prefix provided.

    Args:
        data (pl.DataFrame): The data to be processed.
        dt_prefixes (tuple, optional): A tuple of prefixes for date columns to add month/year columns to. Defaults to ("report", "open").

    Returns:
        pl.DataFrame: The data with additional month and year columns for each specified date prefix.
    """
    data = data.with_columns(
        pl.col(date_labels).dt.month().name.suffix("_month"),
        pl.col(date_labels).dt.year().name.suffix("_year"),
    )
    return data


def adjust_last_dates(data: pl.DataFrame) -> pl.DataFrame:
    """
    Fixes missing pre-last dates and adjusts related columns for consistency.

    Args:
        data (pl.DataFrame): The data to be processed.

    Returns:
        pl.DataFrame: The data with missing pre-last dates fixed and related columns adjusted.
    """
    missing_prelast_mask = (
        get_time_count("report_dt").diff().over(LOAN_IDENTIFIER) > 1
    ) & (pl.col("bal_rur_amt") == 0)
    n_missing_prelast_obs = data.select(missing_prelast_mask.sum()).item(0, 0)
    data = data.with_columns(
        pl.when(missing_prelast_mask)
        .then(pl.col("report_dt").dt.offset_by("-1mo"))
        .otherwise(pl.col("report_dt"))
    )
    logging.info(f"{n_missing_prelast_obs} observations fixed")
    return data


def remove_contracts_with_missing_middle_obs(data: pl.DataFrame) -> pl.DataFrame:
    """
    Removes contracts with missing observations between consecutive dates.

    Args:
        data (pl.DataFrame): The data to be processed.

    Returns:
        pl.DataFrame: The data with contracts having missing observations in the middle removed.
    """
    missing_middle_obs = pl.col("time_count_diff") > 1
    data = data.with_columns(
        has_missing_observations=missing_middle_obs.any().over(LOAN_IDENTIFIER)
    )
    n_contracts_with_missing_observations = (
        data.group_by("has_missing_observations")
        .agg(pl.col(LOAN_IDENTIFIER).n_unique())
        .filter(pl.col("has_missing_observations"))
        .select(LOAN_IDENTIFIER)
        .item(0, 0)
    )
    logging.info(f"{n_contracts_with_missing_observations} contracts removed")
    return data.filter(~pl.col("has_missing_observations")).drop(
        "has_missing_observations"
    )


def deal_with_missing_observations(data: pl.DataFrame) -> pl.DataFrame:
    date_diff = (
        get_time_count("report_dt")
        .diff()
        .over(LOAN_IDENTIFIER)
        .alias("time_count_diff")
    )
    data = data.with_columns(date_diff)
    data = adjust_last_dates(data)
    data = data.with_columns(date_diff)
    data = remove_contracts_with_missing_middle_obs(data)
    return data.drop("time_count_diff")


def transform_date_columns(data: pl.DataFrame) -> pl.DataFrame:
    data = data.drop("start_of_month")
    data = data.with_columns(
        pl.col("report_dt").dt.month_start(),
        pl.col("open_dt").cast(pl.Date).dt.month_start(),
    ).with_columns(
        pl.col("report_dt").min().over(LOAN_IDENTIFIER).name.prefix("min_"),
        pl.col("report_dt").max().over(LOAN_IDENTIFIER).name.prefix("max_"),
        adj_open_dt=pl.min_horizontal(
            pl.col("open_dt", "first_issue_dt").min().over(LOAN_IDENTIFIER)
        ).dt.month_start(),
    )
    return data


def remove_contracts_with_increasing_balance(data: pl.DataFrame) -> pl.DataFrame:
    n_contracts_before = data.select(pl.col(LOAN_IDENTIFIER).n_unique()).item(0, 0)
    data = (
        data.with_columns(
            (pl.col("bal_rur_amt").diff().over(LOAN_IDENTIFIER) > 0).alias(
                "balance_increased"
            )
        )
        .filter(~(pl.col("balance_increased").any().over(LOAN_IDENTIFIER)))
        .drop("balance_increased")
    )
    n_contracts_after = data.select(pl.col(LOAN_IDENTIFIER).n_unique()).item(0, 0)
    logging.info(f"{n_contracts_before - n_contracts_after} contracts removed")
    return data


def remove_contracts_with_zero_balance_for_more_than_one_obs(data):
    n_contracts_before = data.select(pl.col(LOAN_IDENTIFIER).n_unique()).item(0, 0)
    data = (
        data.with_columns(
            ((pl.col("bal_rur_amt") == 0).sum().over(LOAN_IDENTIFIER) > 1).alias(
                "zero_balance_for_more_than_one_obs"
            )
        )
        .filter(~pl.col("zero_balance_for_more_than_one_obs"))
        .drop("zero_balance_for_more_than_one_obs")
    )
    n_contracts_after = data.select(pl.col(LOAN_IDENTIFIER).n_unique()).item(0, 0)
    logging.info(f"{n_contracts_before - n_contracts_after} contracts removed")
    return data


def remove_contracts_with_incomplete_history(data):
    n_contracts_before = data.select(pl.col(LOAN_IDENTIFIER).n_unique()).item(0, 0)
    data = (
        data.with_columns(
            (pl.col("bal_rur_amt") > 0).all().over(LOAN_IDENTIFIER).alias("pos_bal"),
            pl.col("full_prepayment_flag")
            .any()
            .over(LOAN_IDENTIFIER)
            .alias("fp_happened"),
            pl.col("report_dt").max().over(LOAN_IDENTIFIER).alias("max_report_dt"),
        )
        .filter(
            ~pl.col("pos_bal")
            | pl.col("fp_happened")
            | (
                (pl.col("report_dt").max() - pl.col("max_report_dt")).dt.total_days()
                < 365
            )
        )
        .drop("pos_bal", "fp_happened", "max_report_dt")
    )
    n_contracts_after = data.select(pl.col(LOAN_IDENTIFIER).n_unique()).item(0, 0)
    logging.info(f"{n_contracts_before - n_contracts_after} contracts removed")
    return data


def remove_contracts_with_duplicate_observations(data):
    n_contracts_before = data.select(pl.col(LOAN_IDENTIFIER).n_unique()).item(0, 0)
    data = (
        data.with_columns(
            dupes=(pl.col("report_dt").n_unique() != pl.col("report_dt").count()).over(
                LOAN_IDENTIFIER
            )
        )
        .filter(~pl.col("dupes"))
        .drop("dupes")
    )
    n_contracts_after = data.select(pl.col(LOAN_IDENTIFIER).n_unique()).item(0, 0)
    logging.info(f"{n_contracts_before - n_contracts_after} contracts removed")
    return data


def remove_contracts_with_starting_balance_exceeding_loan_amt(data):
    n_contracts_before = data.select(pl.col(LOAN_IDENTIFIER).n_unique()).item(0, 0)
    data = (
        data.with_columns(
            balance_too_big=(pl.col("bal_rur_amt") > pl.col("loan_rur_amt"))
            .any()
            .over(LOAN_IDENTIFIER)
        )
        .filter(~pl.col("balance_too_big"))
        .drop("balance_too_big")
    )
    n_contracts_after = data.select(pl.col(LOAN_IDENTIFIER).n_unique()).item(0, 0)
    logging.info(f"{n_contracts_before - n_contracts_after} contracts removed")
    return data


def add_column_with_initial_value(
    data: pl.DataFrame, value_label: str, drop_old: bool = False, rename: bool = True
) -> pl.DataFrame:
    """
    Adds a column with the initial value for each contract based on another column.

    Args:
        data (pl.DataFrame): The data to be processed.
        value_label (str): The label of the column to use for finding the initial value.
        drop_old (bool, optional): Whether to drop the original column after adding the initial value column. Defaults to False.

    Returns:
        pl.DataFrame: The data with the new column containing the initial value for each contract.
    """
    new_name = f"{value_label}_initial" if rename else value_label
    data = data.with_columns(
        pl.col(value_label).first().over(LOAN_IDENTIFIER).alias(new_name)
    )
    if drop_old:
        data = data.drop(value_label)
    return data


def add_prepayment_flags(data: pl.DataFrame) -> pl.DataFrame:
    """
    Adds a flag indicating full prepayment based on specific advanced repayment types and balance difference.

    Args:
        data (pl.DataFrame): The data to be processed.

    Returns:
        pl.DataFrame: The data with a new column indicating full prepayment.
    """
    data = data.with_columns(
        full_prepayment_flag=(pl.col("advanced_repayment_type") == 2)
        | (pl.col("bal_rur_amt") == 0),
        partial_prepayment_flag=pl.col("advanced_repayment_type") == 1,
    )

    return data


def get_diff_over_window(
    input_column: str,
    over_columns: Union[str, Sequence[str]] = LOAN_IDENTIFIER,
    prefix: str = "",
    postfix: str = "_diff",
    negate: bool = False,
) -> pl.Expr:
    diff_column_name = f"{prefix}{input_column}{postfix}"
    multiplier = -1 if negate else 1
    return (multiplier * pl.col(input_column).diff().over(over_columns)).alias(
        diff_column_name
    )


def add_period_related_columns(data: pl.DataFrame) -> pl.DataFrame:
    data = (
        data.with_columns(
            pl.col("report_dt")
            .cum_count()
            .over(LOAN_IDENTIFIER)
            .alias("obs_period_count"),  # observed period count
        )
        .with_columns(
            true_period_count=(
                pl.col("obs_period_count")
                + get_time_count("min_report_dt")
                - get_time_count("adj_open_dt")
            )
        )
        .with_columns(
            adj_term_residual=(
                pl.col("term_original").first().over(LOAN_IDENTIFIER)
                - pl.col("true_period_count")
            ).clip(lower_bound=0),
        )
    )
    return data


def get_lag_over_window(
    input_columns: Union[str, Sequence[str]],
    over_columns: Union[str, Sequence[str]] = LOAN_IDENTIFIER,
    prefix: str = "",
    postfix: str = "_lag",
    max_lag: int = 1,
) -> List[pl.Expr]:
    if isinstance(input_columns, str):
        input_columns = [input_columns]
    return [
        pl.col(input_column)
        .shift(n_lag)
        .over(over_columns)
        .alias(f"{input_column}_lag{n_lag}")
        for input_column, n_lag in product(input_columns, range(1, max_lag + 1))
    ]


def cast_types(
    data: pl.DataFrame, dtype_columns_mapping: Dict[type, Sequence[str]]
) -> pl.DataFrame:
    column_dtype_mapping = {
        label: dtype
        for dtype, labels in dtype_columns_mapping.items()
        for label in labels
    }
    return data.with_columns(
        pl.col(label).cast(dtype) for label, dtype in column_dtype_mapping.items()
    )


def get_weighted_average(
    value_cols: Union[str, Sequence[str], pl.Expr],
    weights_col: Optional[Union[str, pl.Expr]],
    suffix: str = "_weighted_avg",
    add_suffix=True,
) -> List[pl.Expr]:
    if not isinstance(value_cols, pl.Expr):
        value_cols = pl.col(value_cols)
    if weights_col is None:
        res = value_cols.mean()
    else:
        if isinstance(weights_col, str):
            weights_col = pl.col(weights_col)
        res = (value_cols * weights_col).sum() / weights_col.sum()
    if add_suffix:
        return res.name.suffix(suffix)
    return res


def get_smm(
    balance_label: str, prev_balance_label: str, planned_balance_diff_label: str
) -> pl.Expr:
    smm = 1 - pl.col(balance_label) / (
        pl.col(prev_balance_label) - pl.col(planned_balance_diff_label)
    )
    return smm


def get_cpr(
    smm_label: Optional[str] = None,
    balance_label: Optional[str] = None,
    prev_balance_label: Optional[str] = None,
    planned_balance_diff_label: Optional[str] = None,
):
    if smm_label is not None:
        smm = pl.col(smm_label)
    else:
        smm = get_smm(balance_label, prev_balance_label, planned_balance_diff_label)
    return 1 - (1 - smm) ** 12


def check_classpath_environ():
    if "CLASSPATH" not in os.environ:
        os.environ["CLASSPATH"] = os.popen(
            "$HADOOP_HOME/bin/hadoop classpath --glob "
        ).read()


def load_table_from_hdfs(
    file_name: str,
    convert_datetime_to_date: bool = False,
    cluster_name="",
) -> pl.DataFrame:
    check_classpath_environ()
    hdfs = fs.HadoopFileSystem("hdfs://adhsb")
    data = pq.read_table(file_name, filesystem=hdfs)
    data = pl.from_arrow(data)
    if convert_datetime_to_date:
        data = data.with_columns(
            pl.col(column).cast(pl.Date)
            for column, dtype in zip(data.columns, data.dtypes)
            if isinstance(dtype, pl.Datetime)
        )
    return data


def save_table_to_hdfs(data: pl.DataFrame, file_name: str) -> None:
    check_classpath_environ()
    hdfs = fs.HadoopFileSystem("hdfs://adhsb")
    t = data.to_arrow()
    pq.write_table(t, file_name, filesystem=hdfs)


def get_spark_context(path_to_config: str):
    with open(path_to_config, "r") as f:
        conf_dict = json.load(f)
    conf_list = [(key, value) for key, value in conf_dict.items()]
    conf = SparkConf().setAll(conf_list)
    spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("FATAL")
    return spark


@contextmanager
def spark_context(path_to_config: str) -> SparkSession:
    spark = get_spark_context(path_to_config)
    try:
        yield spark
    finally:
        spark.stop()


def add_product_flags(data: pl.DataFrame) -> pl.DataFrame:
    data = data.with_columns(
        pl.concat_str(
            [
                pl.col("level40_product_bp_nm").fill_null(pl.lit("")),
                pl.col("level50_product_bp_nm").fill_null(pl.lit("")),
            ],
            separator=", ",
        )
        .alias("product")
        .fill_null(pl.lit(""))
    ).with_columns(
        pl.col("product")
        .str.contains(regex_expr)
        .name.suffix(f"_{mortgage_product}_flag")
        for mortgage_product, regex_expr in MORTGAGE_PRODUCT_REGEX_RULES.items()
    )
    product_flag_names = [
        f"product_{mortgage_product}_flag"
        for mortgage_product in MORTGAGE_PRODUCT_REGEX_RULES.keys()
    ]
    return data, product_flag_names


def add_region_flags(
    data: pl.DataFrame, min_n_contracts_in_region: int = 1, common_regions=None
) -> pl.DataFrame:
    if common_regions is None:
        common_regions = (
            (
                data.group_by("region_mid_level")
                .agg(pl.col(LOAN_IDENTIFIER).n_unique().alias("n_contracts"))
                .filter(pl.col("n_contracts") >= min_n_contracts_in_region)
                .select("region_mid_level")
            )
            .transpose()
            .to_numpy()[0]
        )
    region_flag_names = [
        f"region_{region.replace(' ', '_')}_flag" for region in common_regions
    ]
    data = data.with_columns(
        (pl.col("region_mid_level") == region).alias(region_flag_name)
        for region_flag_name, region in zip(region_flag_names, common_regions)
    )
    return data, region_flag_names


def get_aggregated_state_support_flag() -> pl.Expr:
    return pl.any_horizontal(
        f"product_{product}_flag"
        for product in ("state_support", "far_east", "children", "military")
    ).alias("aggregated_state_support_flag")


def add_prepay_amt_columns(data: pl.DataFrame):
    data = (
        data.with_columns(pl.col("prepay_amt").clip(lower_bound=0))
        .with_columns(
            pl.when(pl.col(f"{prefix}_prepayment_flag"))
            .then(pl.col("prepay_amt"))
            .otherwise(0)
            .alias(f"{prefix}_prepay_amt")
            for prefix in ("partial", "full")
        )
        .with_columns(
            prepay_amt=pl.col("partial_prepay_amt") + pl.col("full_prepay_amt")
        )
    )
    return data


def remove_weird_loans(
    data: pl.DataFrame,
):
    ir_conditions = pl.col("interest_rate").is_between(0.01, 0.9).alias("ir_correct")
    loan_amt_conditions = (pl.col("loan_rur_amt") > 0).alias("loan_correct")
    ltv_conditions = pl.col("ltv_initial").is_between(0, 1).alias("ltv_correct")
    data = (
        data.with_columns(ir_conditions, loan_amt_conditions, ltv_conditions)
        .filter(
            pl.all_horizontal(
                pl.col(
                    "ir_correct",
                    "loan_correct",
                    "ltv_correct",
                )
                .all()
                .over(LOAN_IDENTIFIER)
            )
        )
        .drop("ir_correct", "loan_correct", "ltv_correct")
    )

    return data


def prepare_contract_data(
    data: pl.DataFrame, common_regions=None, common_income_segments=None
) -> pl.DataFrame:
    data = data.with_columns(
        pl.col("interest_rate", "ltv_initial").backward_fill().over(LOAN_IDENTIFIER),
        pl.col(pl.Utf8).cast(pl.Categorical),
    ).with_columns(
        pl.when(pl.col("interest_rate") > 1)
        .then(pl.col("interest_rate") / 100)
        .otherwise(pl.col("interest_rate"))
    )
    data = remove_weird_loans(data)
    logging.info("removed weird loans")
    data = data.sort(LOAN_IDENTIFIER, "report_dt")
    logging.info("sorted")
    data = remove_contracts_with_increasing_balance(data)
    logging.info("removed contracts with increasing balance")
    data = remove_contracts_with_starting_balance_exceeding_loan_amt(data)
    logging.info("removed contracts with starting balance exceeding loan amt")
    data = remove_contracts_with_duplicate_observations(data)
    logging.info("removed contracts with duplicate report_dt")
    data = remove_contracts_with_zero_balance_for_more_than_one_obs(data)
    logging.info("removed contracts with zero balance for more than one observation")
    data = add_prepayment_flags(data)
    data = transform_date_columns(data)
    logging.info("transform_date_columns")
    # remove loans with null term_residual
    data = deal_with_missing_observations(data)
    logging.info("deal_with_missing_observations")
    data = add_period_related_columns(data)
    logging.info("add_period_related_columns")

    data = add_bal_diff_prepay_etc(data)
    logging.info("add_bal_diff_prepay_etc")
    data = add_month_year_columns(data, date_labels=["report_dt"])
    logging.info("add_month_year_columns")
    data, product_flag_names = add_product_flags(data)
    logging.info("added product flags")

    data, region_flag_names = add_region_flags(data, common_regions=common_regions)
    logging.info("added region flags")

    data = data.with_columns(get_aggregated_state_support_flag())
    logging.info("added aggregated state support flag")

    data = add_prepay_amt_columns(data)
    logging.info("added prepay amt columns")

    data = data.with_columns(
        pl.col(pl.Categorical).cast(pl.Utf8),
        adj_term_residual_share=pl.col("adj_term_residual")
        / pl.col("adj_term_residual").first().over(LOAN_IDENTIFIER),
    )

    return data


def get_time_count(col_name: str):
    return pl.col(col_name).dt.year() * 12 + pl.col(col_name).dt.month()

