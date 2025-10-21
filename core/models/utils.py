import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
from pyspark import SparkConf
from pyspark.sql import SparkSession, functions
from core.definitions import (
    MATURITY_,
    OPTIONALS_,
    DEFAULT_SEGMENTS_,
    NONDEFAULT_SEGMENTS_,
)


def generate_svo_flg(X):
    svo_flg = X.index == "2022-03-31"
    X["svo_flg"] = svo_flg.astype(float)
    return X


def verify_data(X=None, features_cols=None, y=None, target_cols=None):
    if X is not None:
        features_flag = all(np.isin(features_cols, X.columns))
        if not features_flag:
            raise KeyError(f"X should contains all columns: {features_cols}")

    if y is not None:
        target_flag = all(np.isin(target_cols, y.columns))
        if not target_flag:
            raise KeyError(f"y should contains all columns: {target_cols}")


def dt_convert(dt: datetime) -> str:
    return "".join(str(dt.date()).split("-")[:2])


def check_existence(path, name, overwrite=False):
    return bool((~overwrite) & (name in os.listdir(path)))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def find_left_nearest_point(value, array, is_sorted=False):
    if not is_sorted:
        array = sorted(array)
    for x in array[::-1]:
        if x <= value:
            return x
    return array[0]


def gen_opt_model_name(segment):
    if segment is None:
        segment = "segment"
    elif segment not in ["mass", "priv", "svip", "bvip"]:
        raise ValueError(
            "segment should be like ['mass', 'priv', 'svip', 'bvip'] or None"
        )
    return f"opt_structure_{segment}"


def gen_newbiz_model_name(segment):
    if segment is None:
        segment = "segment"
    elif segment not in ["mass", "priv", "svip", "bvip"]:
        raise ValueError(
            "segment should be like ['mass', 'priv', 'svip', 'bvip'] or None"
        )
    return f"newbusiness_{segment}"


# Функции для генерации имени модели для накопительных счетов
def gen_sa_product_balance_model_name(product=None, segment=None):
    if product is None:
        product = "general"
    elif product not in ["classic", "kopilka", "general"]:
        raise ValueError(
            "segment should be like ['classic', 'kopilka', 'general'] or None"
        )

    if segment is None:
        segment = "segment"
    elif segment not in ["mass", "priv", "vip"]:
        raise ValueError("segment should be like ['mass', 'priv', 'vip'] or None")
    return f"sa_{product}_avg_balance_{segment}"


def gen_sa_product_structure_model_name(segment):
    if segment is None:
        segment = "segment"
    elif segment not in ["mass", "priv", "vip"]:
        raise ValueError("segment should be like ['mass', 'priv', 'vip'] or None")
    return f"sa_product_structure_{segment}"


def run_spark_session(name: str = "session") -> SparkSession:
    SPARK_CONFIG = [
        # Driver
        ("spark.driver.cores", "8"),
        ("spark.driver.memory", "32g"),
        ("spark.driver.maxResultSize", "64g"),
        # Executor
        ("spark.executor.cores", "8"),
        ("spark.executor.memory", "16g"),
        # Seriolization & Arrow
        ("spark.kryoserializer.buffer.max", "2000m"),
        ("spark.sql.execution.arrow.enabled", "true"),  # speeds up .toPandas()
        ("spark.sql.execution.arrow.pyspark.enabled", "true"),
        # Dynamic Allocation
        ("spark.dynamicAllocation.enabled", "True"),
        ("spark.dynamicAllocation.initialExecutors", "2"),
        ("spark.dynamicAllocation.maxExecutors", "32"),
        # Timezone
        ("spark.sql.session.timeZone", "Europe/Moscow"),
    ]

    conf = SparkConf().setAll(SPARK_CONFIG)

    spark = (
        SparkSession.builder.master("yarn")
        .appName(f"{name}")
        .config(conf=conf)
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    return spark


def dec_detect(col: str, col_type: str):
    dec_flag = col_type[:8] == "decimal("
    decint_flag = col_type[-3:] == ",0)"
    if dec_flag:
        if decint_flag:
            return functions.col(col).cast("integer")
        else:
            return functions.col(col).cast("double")
    else:
        return functions.col(col)


def convert_decimals(spark_df):
    select_expr = [dec_detect(c, t) for c, t in spark_df.dtypes]
    return spark_df.select(select_expr)


def get_feature_name(
    feature,
    segment: str = None,
    repl: int = None,
    sub: int = None,
    maturity: int = None,
):
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


def calculate_weighted_rates(
    df: pd.DataFrame, segment: str = None, repl: int = None, sub: int = None
):
    if (repl is not None) and (sub is not None):
        rates = [
            get_feature_name("VTB_weighted_rate", segment, repl, sub, mat)
            for mat in MATURITY_
        ]
        weights = [
            get_feature_name("y_inflow_share", segment, repl, sub, mat)
            for mat in MATURITY_
        ]
        return np.sum(df[rates].values * df[weights].values, axis=1).reshape(-1, 1)
    if (repl is not None) or (sub is not None):
        raise ValueError("Calculation only for repl or sub are not supported yet")
    if segment is not None:
        rates = [
            get_feature_name("VTB_weighted_rate", segment, repl, sub)
            for repl, sub in OPTIONALS_
        ]
        weights = [
            get_feature_name("y_inflow_share", segment, repl, sub)
            for repl, sub in OPTIONALS_
        ]
        if all(np.isin(weights, df.columns)):
            return np.sum(df[rates].values * df[weights].values, axis=1).reshape(-1, 1)
        elif segment == "vip":
            weights = [
                get_feature_name("y_inflow_share", segment="svip", repl=repl, sub=sub)
                for repl, sub in OPTIONALS_
            ]
            return np.sum(df[rates].values * df[weights].values, axis=1).reshape(-1, 1)
        else:
            raise KeyError(f"{weights} not in dataframe columns")
    else:
        rates = [
            get_feature_name("VTB_weighted_rate", seg)
            for seg in ["mass", "priv", "vip"]
        ]
        weights = [
            get_feature_name("y_inflow_share", seg) for seg in ["mass", "priv", "vip"]
        ]
        if all(np.isin(weights, df.columns)):
            return np.sum(df[rates].values * df[weights].values, axis=1).reshape(-1, 1)
        else:
            weights = [
                get_feature_name("y_inflow", seg) for seg in ["mass", "priv", "vip"]
            ]
            return np.sum(
                df[weights].values
                / df[weights].sum(axis=1).values.reshape(-1, 1)
                * df[rates].values,
                axis=1,
            ).reshape(-1, 1)


def calculate_max_rate(df, segment=None, repl=None, sub=None):
    if (repl is not None) and (sub is not None):
        rates = [
            get_feature_name("VTB_weighted_rate", segment, repl, sub, mat)
            for mat in MATURITY_
        ]
        return df[rates].max(axis=1).values.reshape(-1, 1)
    if (repl is not None) or (sub is not None):
        raise ValueError("Calculation only for repl or sub are not supported yet")
    if segment is not None:
        rates = [
            get_feature_name("VTB_weighted_rate", segment, repl, sub, mat)
            for repl, sub in OPTIONALS_
            for mat in MATURITY_
        ]
        return df[rates].max(axis=1).values.reshape(-1, 1)
    else:
        rates = [
            get_feature_name("VTB_weighted_rate", segment, repl, sub, mat)
            for segment in DEFAULT_SEGMENTS_
            for repl, sub in OPTIONALS_
            for mat in MATURITY_
        ]
        return df[rates].max(axis=1).values.reshape(-1, 1)


def calculate_max_available_rate(df, segment, repl=None, sub=None, mat=None):
    available_segments_map = {
        "mass": ["mass"],
        "priv": ["mass", "priv"],
        "vip": ["mass", "priv", "vip"],
    }
    available_segments = available_segments_map[segment]

    if (mat is not None) and (repl is not None) and (sub is not None):
        rates = [
            get_feature_name("VTB_weighted_rate", segment, repl, sub, mat)
            for segment in available_segments
        ]
        return df[rates].max(axis=1).values.reshape(-1, 1)
    if (repl is not None) and (sub is not None):
        rates = [
            get_feature_name("VTB_weighted_rate", segment, repl, sub, mat)
            for segment in available_segments
            for mat in MATURITY_
        ]
        return df[rates].max(axis=1).values.reshape(-1, 1)
    if (repl is not None) or (sub is not None):
        raise ValueError("Calculation only for repl or sub are not supported yet")
    if segment is not None:
        rates = [
            get_feature_name("VTB_weighted_rate", segment, repl, sub, mat)
            for segment in available_segments
            for repl, sub in OPTIONALS_
            for mat in MATURITY_
        ]
        return df[rates].max(axis=1).values.reshape(-1, 1)
    else:
        rates = [
            get_feature_name("VTB_weighted_rate", segment, repl, sub, mat)
            for segment in DEFAULT_SEGMENTS_
            for repl, sub in OPTIONALS_
            for mat in MATURITY_
        ]
        return df[rates].max(axis=1).values.reshape(-1, 1)


def calculate_weighted_ftp_rate(df: pd.DataFrame, segment: str = None):
    if segment is not None:
        cur_rate = 0
        rates = [get_feature_name("VTB_ftp_rate", maturity=mat) for mat in MATURITY_]
        for repl, sub in OPTIONALS_:
            weights = [
                get_feature_name("y_inflow_share", segment, repl, sub, mat)
                for mat in MATURITY_
            ]
            opt_share = get_feature_name("y_inflow_share", segment, repl, sub)
            if opt_share not in df.columns and segment == "vip":
                opt_share = get_feature_name("y_inflow_share", "svip", repl, sub)
            cur_rate += np.sum(
                df[weights].values * df[[opt_share]].values * df[rates].values, axis=1
            )
        return cur_rate.reshape(-1, 1)
    else:
        raise ValueError("Calculation only for segment None are not supported yet")


def calculate_max_weighted_rate(
    df: pd.DataFrame, segment: str = None, repl: int = None, sub: int = None
):
    if (repl is not None) and (sub is not None):
        return calculate_max_rate(df, segment, repl, sub)
    if (repl is not None) or (sub is not None):
        raise ValueError("Calculation only for repl or sub is not supported yet")
    if segment is not None:
        mat_weights_in_opt = {}
        absolute_weights = np.zeros(shape=(df.shape[0], len(MATURITY_)))
        absolute_weights_prod_rate = np.zeros(shape=(df.shape[0], len(MATURITY_)))
        for repl, sub in OPTIONALS_:
            rates = [
                get_feature_name("VTB_weighted_rate", segment, repl, sub, maturity=mat)
                for mat in MATURITY_
            ]
            weights = [
                get_feature_name("y_inflow_share", segment, repl, sub, mat)
                for mat in MATURITY_
            ]
            opt_share = get_feature_name("y_inflow_share", segment, repl, sub)
            if opt_share not in df.columns and segment == "vip":
                opt_share = get_feature_name("y_inflow_share", "svip", repl, sub)
            absolute_weights += df[weights].values * df[[opt_share]].values
            absolute_weights_prod_rate += (
                df[weights].values * df[[opt_share]].values * df[rates].values
            )
        return np.max(absolute_weights_prod_rate / absolute_weights, axis=1).reshape(
            -1, 1
        )


def calculate_absolute_inflows_default_segments(df):
    res = pd.DataFrame(index=df.index)
    for segment in DEFAULT_SEGMENTS_:
        y_inflow_abs = df.loc[:, [get_feature_name("y_inflow", segment)]].values
        for repl, sub in OPTIONALS_:
            y_opt_share = df.loc[
                :, [get_feature_name("y_inflow_share", segment, repl, sub)]
            ].values
            y_opt_abs_name = get_feature_name("y_inflow", segment, repl, sub)
            y_opt_abs = y_inflow_abs * y_opt_share
            res.loc[:, y_opt_abs_name] = y_opt_abs
            for mat in MATURITY_:
                y_opt_mat_share = df.loc[
                    :, [get_feature_name("y_inflow_share", segment, repl, sub, mat)]
                ].values
                y_opt_mat_abs_name = get_feature_name(
                    "y_inflow", segment, repl, sub, mat
                )
                y_opt_mat_abs = y_opt_abs * y_opt_mat_share
                res.loc[:, y_opt_mat_abs_name] = y_opt_mat_abs
                return res


# Эту функцию я писал в состоянии полусмерти
def calculate_absolute_inflows_nondefault_segments(df):
    res = pd.DataFrame(index=df.index)
    res.loc[:, get_feature_name("y_inflow")] = df.loc[
        :, [get_feature_name("y_inflow", segment) for segment in NONDEFAULT_SEGMENTS_]
    ].sum(axis=1)
    for segment in NONDEFAULT_SEGMENTS_:
        y_inflow_abs_name = get_feature_name("y_inflow", segment)
        y_inflow_abs = df.loc[:, [y_inflow_abs_name]].values
        res.loc[:, y_inflow_abs_name] = y_inflow_abs
        for repl, sub in OPTIONALS_:
            y_opt_share = df.loc[
                :, [get_feature_name("y_inflow_share", segment, repl, sub)]
            ].values
            y_opt_abs_name = get_feature_name("y_inflow", segment, repl, sub)
            y_opt_abs = y_inflow_abs * y_opt_share
            res.loc[:, y_opt_abs_name] = y_opt_abs
    y_inflow_abs_vip_name = get_feature_name("y_inflow", "vip")
    res.loc[:, y_inflow_abs_vip_name] = (
        res.loc[:, [get_feature_name("y_inflow", "svip")]].values
        + res.loc[:, [get_feature_name("y_inflow", "bvip")]].values
    )
    y_opt_abs_vip_names = [
        get_feature_name("y_inflow", "vip", repl, sub) for repl, sub in OPTIONALS_
    ]
    res.loc[:, y_opt_abs_vip_names] = (
        res.loc[
            :,
            [
                get_feature_name("y_inflow", "svip", repl, sub)
                for repl, sub in OPTIONALS_
            ],
        ].values
        + res.loc[
            :,
            [
                get_feature_name("y_inflow", "bvip", repl, sub)
                for repl, sub in OPTIONALS_
            ],
        ].values
    )
    for segment in DEFAULT_SEGMENTS_:
        for repl, sub in OPTIONALS_:
            y_opt_mat_abs_names = [
                get_feature_name("y_inflow", segment, repl, sub, mat)
                for mat in MATURITY_
            ]
            res.loc[:, y_opt_mat_abs_names] = (
                res.loc[:, [get_feature_name("y_inflow", segment, repl, sub)]].values
                * df.loc[
                    :,
                    [
                        get_feature_name("y_inflow_share", segment, repl, sub, mat)
                        for mat in MATURITY_
                    ],
                ].values
            )
    return res


def calculate_sa_model_features(df: pd.DataFrame, features):
    sa_features_map = {
        "SA_weighted_rate_[general]": lambda segment: df.loc[
            :, "rate_sa_weighted"
        ].values.reshape(-1, 1),
        "DPST_rate_[long]": lambda segment: np.concatenate(
            [
                calculate_max_available_rate(df, segment, repl, sub, mat)
                for repl, sub in OPTIONALS_
                for mat in MATURITY_
                if mat >= 365
            ],
            axis=1,
        )
        .max(axis=1)
        .reshape(-1, 1),
        "DPST_rate_[general]": lambda segment: df.loc[
            :, get_feature_name("VTB_weighted_rate", segment)
        ].values.reshape(-1, 1),
        "DPST_rate_[sber]": lambda segment: df.loc[:, "SBER_max_rate"].values.reshape(
            -1, 1
        ),
        "DPST_plan_close": lambda segment: df.loc[
            :, get_feature_name("plan_close", segment)
        ].values.reshape(-1, 1),
    }

    feature_df = pd.DataFrame(index=df.index)
    for feature in features:
        segment = re.search("|".join(DEFAULT_SEGMENTS_), feature)
        if segment:
            segment = segment[0]
            key = re.sub(f"\[{segment}\]", "", feature).strip("_")
        else:
            key = feature

        feature_df.loc[:, feature] = sa_features_map[key](segment)
    return feature_df


# для моделей долей по бакетам
def gaussian_kernel(X, sigma):
    distances = abs(X[0] - X[1])
    kernel_matrix = np.exp(-distances / (2 * sigma**2))

    return kernel_matrix


def calc_model_bucket_share(spread=0):
    """
    считает перетоки между бакетами балагнса при изменении спреда. Возвращает долю которая перетечет в бакет выше
    spread - %, спред к ближайшему бакету

    share - return, возвращает долю перетока в данном бакете в более крупный бакет
    """

    if spread < 0.01:
        return 0

    y = []

    for i in np.arange(0, 0.51, 0.01):
        y.append(gaussian_kernel([i, 0.5], np.tanh(min(0.02 + 0.2 * spread, 0.25))))

    share = (np.array(y) * 0.01 * 2).sum()

    return share


def calc_new_shares(parse_res, BALANCE_BUCKETS, segment, features):
    """
    Функция считает доли по новому распределению
    """

    parse_res_new = parse_res.copy()

    for i in range(len(BALANCE_BUCKETS) - 1):
        # откуда
        bucket0 = BALANCE_BUCKETS[i]
        # куда
        bucket1 = BALANCE_BUCKETS[i + 1]

        feature_name = f"spread_VTB_rate_[{segment}]_[{bucket1}-{bucket0}]"

        share_flows = calc_model_bucket_share(spread=features[feature_name].values[0])

        add_flow = parse_res_new[bucket0] * share_flows

        parse_res_new[bucket0] = round(parse_res_new[bucket0] - add_flow, 4)
        parse_res_new[bucket1] = round(parse_res_new[bucket1] + add_flow, 4)

    return parse_res_new


def parse_buckets_from_port(port, segment, balance_buckets):
    """
    Функция которая парсит данные с портфеля

    port - портфель
    segment - mass, priv, vip
    balance_buckets - используемые бакеты баланса

    """

    DEFAULT_SEGMENTS_MAP_ = {"mass": 0, "priv": 1, "vip": 2}

    # выделяем цифру сегмента и фильтруем
    segm_num = DEFAULT_SEGMENTS_MAP_[segment]
    share_buckets_str = port[
        port["is_vip_or_prv"] == segm_num
    ].share_buckets_balance.max()

    # заполняем бакеты баланса
    dict_res = {}
    share_buckets_str = share_buckets_str.split(",")
    for share in share_buckets_str:
        name, share = share.split(")_")

        dict_res[name + ")"] = float(share)

    return dict_res
