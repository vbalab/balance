import os.path
from datetime import datetime, timedelta
from functools import reduce
from itertools import product
from bisect import bisect_left
from pickle import PickleError, dump
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np
import pyspark.sql.functions as f
from pyspark.sql import Window
from pyspark.sql.functions import when
from dateutil.relativedelta import relativedelta
from pandas import DataFrame
from pyspark.sql import Window
from pyspark.sql.dataframe import DataFrame as spDataFrame
from core.upfm.commons import (
    _REPORT_DT_COLUMN,
    BaseModel,
    DataLoader,
    ModelTrainer,
    ForecastContext,
    ModelInfo,
)
from .base_model import EarlyRedemptionModel, early_redemption_ptr_transform
from core.models.utils import dt_convert, find_left_nearest_point
from core.definitions import (
    DEFAULT_SEGMENTS_,
    OPTIONALS_,
    MATURITY_,
    get_feature_name,
    PORTFOLIO_COLUMNS_,
    MONTH_MATURITY_MAP_,
    MATURITY_TO_MONTH_MAP_,
)
from .spark_pandas import spark_df_toPandas
from pandas.tseries.offsets import MonthEnd

# try:
#     from deposit_early_redemption.base_model import (
#         EarlyRedemptionModel, early_redemption_ptr_transform)
#     from deposit_early_redemption.spark_pandas import spark_df_toPandas
# except ModuleNotFoundError:
#     from base_model import EarlyRedemptionModel, early_redemption_ptr_transform
#     from spark_pandas import spark_df_toPandas


_DEFAULT_START_DATE_NOOPT = datetime(2014, 1, 31)
_DEFAULT_HYPERPARAMS_NOOPT = dict(
    n_iter=2,  # число итераций дерева
    p_step=2,  # шаг перцентиля при выборе трешхолда
    p_low=2,  # нижний перцентить при выборе трешхолда
    p_high=98,  # верхний перцентить при выборе трешхолда
    add_constant=False,  # на самом деле она и так добавляется, лишний параметр приведет к ошибке
    scaler=False,  # лучше не менять на Тру, т.к. не был протестирвоан этот вариант
    entity_effects=False,  # Сильно замедляет обучение, не сильно улучшает результаты
    lasso_alpha=None,  # приведет к оишбке изменение
    rank_check_in_iters=True,  # проверка ранга при переборе трешхолда
)

_DEFAULT_COLS = dict(
    target="SER_dinamic_cl",
    leaf_reg_features=["incentive", "incentive_lag1", "share_period_plan"],
    splitting_cols=["share_period_plan", "incentive", "months_left", "incentive_lag1"],
    splitting_cols_labels=["shr", "inc", "lft", "lag"],
    weight_col="total_generation_cl_lag1",
)

# TODO: Исправить под современные реалии
# EXTRENAL_MODELS_CONFIGS = {
#     #'maturity_buckets': ['1-3m', '4-6m', '7-12m', '13-18m', '19-24m', '25-36m', '36+m'],
#     #'size_buckets': ['100k', '400k', '1000k', '2000k', '2000kplus'],
#     'configs': [
#         {
#             'optionality': ['R0S0'], 'segment': ['mass'], 'cur': ['RUR'],
#             'maturity_buckets': ['1-3m', '4-6m', '7-12m', '13-18m', '19-24m', '25-36m', '36+m'],
#             'maturity_buckets_int': [3, 6, 12, 18, 24, 36, 48],
#             'size_buckets': ['100k', '400k', '1000k', '2000k', '2000kPlus'],
#             'size_buckets_int': [100, 400, 1_000, 2_000, 3_000],
#             'scenario_vtb_rates_prefix': 'RUB.DEPOSITS.NOVIP.NOOPT.VTB.RATE',
#             'maturity_structure_first_model_key': 'MATURITY_STRUCTURE',
#             'maturity_structure_second_model_key': 'NOVIP_NOOPT',
#             'maturity_structure_col_prefix': 'RUB.DEPOSITS.NOVIP.NOOPT.VTB.SHARE',
#             'size_structure_first_model_key': 'SIZE_STRUCTURE',
#             'size_structure_second_model_key': 'NOVIP_NOOPT',
#             'size_structure_col_prefix': 'RUB.DEPOSITS.NOVIP.NOOPT.VTB.SHARE',
#             'newbusiness_first_model_key': 'NEWBUSINESS',
#             'newbusiness_second_model_key': 'NOVIP_NOOPT',
#             'newbusiness_col_name': 'RUB.DEPOSITS.NOVIP.NOOPT.VTB.NEWBUSINESS',
#         },
#         {
#             'optionality': ['R0S0'], 'segment': ['vip'], 'cur': ['RUR'],
#             'maturity_buckets': ['1-3m', '4-6m', '7-12m', '13-18m', '19-24m', '25-36m', '36+m'],
#             'maturity_buckets_int': [3, 6, 12, 18, 24, 36, 48],
#             'size_buckets': ['100k', '400k', '1000k', '2000k', '2000kPlus'],
#             'size_buckets_int': [100, 400, 1_000, 2_000, 3_000],
#             'scenario_vtb_rates_prefix': 'RUB.DEPOSITS.VIP.NOOPT.VTB.RATE',
#             'maturity_structure_first_model_key': 'MATURITY_STRUCTURE',
#             'maturity_structure_second_model_key': 'VIP_NOOPT',
#             'maturity_structure_col_prefix': 'RUB.DEPOSITS.VIP.NOOPT.VTB.SHARE',
#             'size_structure_first_model_key': 'SIZE_STRUCTURE',
#             'size_structure_second_model_key': 'VIP_NOOPT',
#             'size_structure_col_prefix': 'RUB.DEPOSITS.VIP.NOOPT.VTB.SHARE',
#             'newbusiness_first_model_key': 'NEWBUSINESS',
#             'newbusiness_second_model_key': 'VIP_NOOPT',
#             'newbusiness_col_name': 'RUB.DEPOSITS.VIP.NOOPT.VTB.NEWBUSINESS',
#         },
#         {
#             'optionality': ['opt'], 'segment': ['novip'], 'cur': ['RUR'],
#             'maturity_buckets': ['1-3m', '4-6m', '7-12m', '13-18m', '19-24m', '25-36m', '36+m'],
#             'maturity_buckets_int': [3, 6, 12, 18, 24, 36, 48],
#             'size_buckets': ['10k', '100k', '400k', '1000k', '1400k', '1400kPlus'],
#             'size_buckets_int': [10, 100,    400,   1_000,    1_400,   2_000],
#             'scenario_vtb_rates_prefix': 'RUB.DEPOSITS.NOVIP.OPT.VTB.RATE',
#             'maturity_structure_first_model_key': 'MATURITY_STRUCTURE',
#             'maturity_structure_second_model_key': 'NOVIP_OPT',
#             'maturity_structure_col_prefix': 'RUB.DEPOSITS.NOVIP.OPT.VTB.SHARE',
#             'size_structure_first_model_key': 'SIZE_STRUCTURE',
#             'size_structure_second_model_key': 'NOVIP_OPT',
#             'size_structure_col_prefix': 'RUB.DEPOSITS.NOVIP.OPT.VTB.SHARE',
#             'newbusiness_first_model_key': 'NEWBUSINESS',
#             'newbusiness_second_model_key': 'NOVIP_OPT',
#             'newbusiness_col_name': 'RUB.DEPOSITS.NOVIP.OPT.VTB.NEWBUSINESS',
#         },
#         {
#             'optionality': ['opt'], 'segment': ['vip'], 'cur': ['RUR'],
#             'maturity_buckets': ['1-3m', '4-6m', '7-12m', '13-18m', '19-24m', '25-36m', '36+m'],
#             'maturity_buckets_int': [3, 6, 12, 18, 24, 36, 48],
#             'size_buckets': ['10k', '100k', '400k', '1000k', '1400k', '1400kPlus'],
#             'size_buckets_int': [10, 100,    400,   1_000,    1_400,   2_000],
#             'scenario_vtb_rates_prefix': 'RUB.DEPOSITS.VIP.OPT.VTB.RATE',
#             'maturity_structure_first_model_key': 'MATURITY_STRUCTURE',
#             'maturity_structure_second_model_key': 'VIP_OPT',
#             'maturity_structure_col_prefix': 'RUB.DEPOSITS.VIP.OPT.VTB.SHARE',
#             'size_structure_first_model_key': 'SIZE_STRUCTURE',
#             'size_structure_second_model_key': 'VIP_OPT',
#             'size_structure_col_prefix': 'RUB.DEPOSITS.VIP.OPT.VTB.SHARE',
#             'newbusiness_first_model_key': 'NEWBUSINESS',
#             'newbusiness_second_model_key': 'VIP_OPT',
#             'newbusiness_col_name': 'RUB.DEPOSITS.VIP.OPT.VTB.NEWBUSINESS',
#         }
#     ]
# }


CONFIG = {
    "common_params": {
        "class_name_prefix": "EarlyRedemption",
        "file_name_prefix": "deposit_earlyredemption",
        "dbase_name": "dadm_alm_sbx.deposits_early_close_v2",
        "report_date_col": "report_date",
        "date_format": "%Y-%m",
        "filter_attributes": ["segment", "optionality", "cur"],  # dbase filters
        # this maps segment/optionality/cur to the corresponding col of dbase_name and its values
        "segment_map": {
            "mass": {"is_vip_or_prv": [0]},
            "priv": {"is_vip_or_prv": [1]},
            "vip": {"is_vip_or_prv": [2]},
        },
        "optionality_map": {
            "R0S0": {"optional_flg": [0]},
            "R0S1": {"optional_flg": [1]},
            "R1S0": {"optional_flg": [2]},
            "R1S1": {"optional_flg": [3]},
        },
        "cur_map": {
            "RUR": {"cur": ["RUR"]},
            "USD": {"cur": ["USD"]},
            "EUR": {"cur": ["EUR"]},
        },
        # Пофиксить это
        "portfolio_key": "portfolio",
        "features_key": "features",  # key in model_data'
        "portfolio_filter_attrs": [
            "optionality",
            "segment",
        ],  # filet atrrs on model_data portfolio
        "periods_list": [1, 3, 6, 12, 24],
        "rate_buckets": list(range(100)),
        "balance_buckets": [1, 2, 3, 4, 5],
    },
    "models_params": [
        {
            "segment": [segment],
            "optionality": [f"R{repl}S{sub}"],
            "replenishable_flg": repl,
            "subtraction_flg": sub,
            "cur": ["RUR"],
            "hyperparams": _DEFAULT_HYPERPARAMS_NOOPT,
            "start_date": _DEFAULT_START_DATE_NOOPT,
            "default_start_date": _DEFAULT_START_DATE_NOOPT,
            "model_class": EarlyRedemptionModel,
            "model_class_kwargs": _DEFAULT_COLS,
            "scenario_vtb_rates_cols": [
                get_feature_name("VTB_weighted_rate", segment, repl, sub, mat)
                for mat in MATURITY_
                if mat < 1000
            ],
            "weighted_vtb_rate": get_feature_name(
                "VTB_weighted_rate", segment, repl, sub
            )
            # Выяснить что это значит
            #'external_configs': EXTRENAL_MODELS_CONFIGS['configs'][:2]
        }
        for segment in DEFAULT_SEGMENTS_
        for repl, sub in OPTIONALS_
    ],
}


def gen_file_name(config, model_params):
    prefix = config["common_params"]["file_name_prefix"]
    parts = [prefix]
    for attr in config["common_params"]["filter_attributes"]:
        parts.append("_".join(model_params[attr]))
    return "_".join(parts)


def gen_class_name(config, model_params, model_type):
    prefix = config["common_params"]["class_name_prefix"]
    parts = [prefix]
    for attr in config["common_params"]["filter_attributes"]:
        part = "".join(s[0].upper() + s[1:] for s in model_params[attr])
        parts.append(part)
    parts.append(model_type)
    return "".join(parts)


def gen_where_cond(attribute: str, cls):
    attr_params: List[str] = getattr(cls, attribute)
    attr_map: Dict[str, Dict[str, List[str]]] = getattr(cls, attribute + "_map")
    conds = []
    for param_value in attr_params:
        col_value_map = attr_map[param_value]
        cond = reduce(
            lambda x, y: (x & y),
            [f.col(col).isin(val) for col, val in col_value_map.items()],
        )
        conds.append(cond)
    return reduce(lambda x, y: (x | y), conds)


def gen_conds(attributes: List[str], cls):
    conds = [gen_where_cond(attribute, cls) for attribute in attributes]
    return reduce(lambda x, y: (x & y), conds)


class Meta(type):
    def __new__(cls, class_name, abc_base, attrs):
        if not attrs:
            attrs = dict()

        exclude_attrs = {
            "__module__",
            "__dict__",
            "__weakref__",
            "__abstractmethods__",
            "_abc_impl",
        }
        abc_attrs = list(set(abc_base.__dict__.keys()) - exclude_attrs)

        for m in abc_attrs:
            attrs[m] = getattr(abc_base, m)

        for m in [m for m in dir(cls) if not m.startswith("__")]:
            attrs[m] = getattr(cls, m)
            if m in abc_base.__abstractmethods__:
                attrs[m].__doc__ = getattr(abc_base, m).__doc__

        attrs["abc_base"] = abc_base

        for m in abc_base.__abstractmethods__:
            if not attrs.get(m, False):
                raise TypeError(
                    "Can't instantiate abstract class "
                    + f"{class_name} with abstract methods {m}"
                )
            else:
                attrs[m].__doc__ = getattr(abc_base, m).__doc__
        """
        
        for m in [m for m in dir(abc_base) if not m.startswith('__')]:
            if m in dir(cls):
                pass
            else:
                attrs[m] = getattr(abc_base, m)
        """

        return super().__new__(cls, class_name, (), attrs)

    def __init__(self, model_params, abc_base):
        super().__init__(self.class_name, (abc_base,), dict())


class MetaDataLoader(Meta):
    def __new__(cls, model_params, abc_base):
        attrs = dict()
        attrs["model_type"] = "DataLoader"
        attrs["class_name"] = gen_class_name(CONFIG, model_params, attrs["model_type"])
        attrs["file_name"] = gen_file_name(CONFIG, model_params)
        attrs["date_format"] = "%Y-%m"

        attrs.update(model_params)
        attrs.update(CONFIG["common_params"])

        return super().__new__(cls, attrs["class_name"], abc_base, attrs)

    def _date_extractor(self, string: str) -> datetime:
        init = datetime.strptime(string, self.date_format)
        return init + relativedelta(months=1) - timedelta(days=1)

    def _gen_where_cond(self, attribute: str):
        attr_params: List[str] = getattr(self, attribute)
        attr_map: Dict[str, Dict[str, List[str]]] = getattr(self, attribute + "_map")
        conds = []
        for param_value in attr_params:
            col_value_map = attr_map[param_value]
            cond = reduce(
                lambda x, y: (x & y),
                [f.col(col).isin(val) for col, val in col_value_map.items()],
            )
            conds.append(cond)
        return reduce(lambda x, y: (x | y), conds)

    def _gen_conds(self, attributes: List[str]):
        conds = [self._gen_where_cond(attribute) for attribute in attributes]
        return reduce(lambda x, y: (x & y), conds)

    def get_maximum_train_range(self, spark) -> Tuple[datetime, datetime]:
        a = spark.sql(
            f"""
                      select open_month
                      from {self.dbase_name}
                      """
        ).filter(self._gen_conds(self.filter_attributes))
        a = a.agg(
            f.min(f.col("open_month")).alias("min"),
            f.max(f.col("open_month")).alias("max"),
        ).collect()

        min_open_month = self._date_extractor(a[0].asDict()["min"])
        max_open_month = self._date_extractor(a[0].asDict()["max"])

        return max(min_open_month, self.start_date), max_open_month

    @staticmethod
    def _gen_weird_flag(pd_df: DataFrame) -> DataFrame:
        close_dt = pd.to_datetime(
            pd_df.close_month, errors="coerce"
        ) + pd.offsets.MonthEnd(1)

        flags = [
            (pd_df["max_SER_dinamic"] > 5),
            (pd_df["max_total_generation"] <= 100_000),
            (pd_df["drop_flg"] != 0),
            (pd_df["total_generation_cleared"].isna()),
            (
                # не совсем понятные условия
                True
                & ((close_dt - pd_df[_REPORT_DT_COLUMN]).dt.days < 33)
                & (pd_df["SER_dinamic"] < -0.05)
                & (
                    pd_df["report_weight_open_rate_24m"]
                    > pd_df["report_weight_open_rate_3m"]
                )
            ),
        ]

        frame = pd_df[["gen_name"]].copy()
        frame["weird_flag"] = reduce(lambda x, y: x | y, flags)

        return frame.groupby("gen_name")["weird_flag"].transform("max")

    @staticmethod
    def _gen_weird_flag_spark(df):
        #         close_dt = pd.to_datetime(pd_df.close_month, errors = 'coerce') + pd.offsets.MonthEnd(1)

        #         flags = [
        #             (pd_df['max_SER_dinamic'] > 5),
        #             (pd_df['max_total_generation'] <= 100_000),
        #             (pd_df['drop_flg'] != 0),
        #             (pd_df['total_generation_cleared'].isna()),
        #             (
        #                 # не совсем понятные условия
        #                 True
        #                 & ((close_dt - pd_df[_REPORT_DT_COLUMN]).dt.days < 33)
        #                 & (pd_df['SER_dinamic'] < -0.05)
        #                 & (pd_df['report_weight_open_rate_24m'] > pd_df['report_weight_open_rate_3m'])
        #             )
        #         ]

        #         frame = pd_df[['gen_name']].copy()
        #         frame['weird_flag'] = reduce(lambda x, y: x | y, flags)

        df = df.withColumn("close_dt", f.last_day(f.to_date("close_month")))

        select_cols = [
            "close_dt",
            "max_SER_dinamic",
            "max_total_generation",
            "drop_flg",
            "total_generation_cleared",
            _REPORT_DT_COLUMN,
            "SER_dinamic",
            "report_weight_open_rate_24m",
            "report_weight_open_rate_3m",
            "gen_name",
        ]

        # создаем глубокую копию в pySpark
        df_weird = df.select(*select_cols)

        df_weird = df_weird.withColumn(
            "weird_flag",
            when(
                (df_weird.max_SER_dinamic > 5)
                | (df_weird.max_total_generation <= 100000)
                | (df_weird.drop_flg != 0)
                | (f.isnull(df_weird.total_generation_cleared))
                | (
                    (f.datediff(df_weird.close_dt, _REPORT_DT_COLUMN) < 33)
                    & (df_weird.SER_dinamic < -0.05)
                    & (
                        df_weird.report_weight_open_rate_24m
                        > df_weird.report_weight_open_rate_3m
                    )
                ),
                1,
            ).otherwise(0),
        )

        df_weird = df_weird.select("gen_name", "weird_flag")

        df_weird = df_weird.groupBy("gen_name").agg(
            f.max("weird_flag").alias("weird_flag")
        )

        return df_weird

    def _get_basic_portfolios(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> spDataFrame:
        sql = f"""
        select * from {self.dbase_name}
        where 1=1
            and close_month > report_month
            -- and drop_flg = 0
            and report_month <= '{end_date.strftime(self.date_format)}' 
            and report_month >= '{start_date.strftime(self.date_format)}'
        """
        w = Window.partitionBy("gen_name")

        data_sp = (
            spark.sql(sql)
            .withColumnRenamed(self.report_date_col, _REPORT_DT_COLUMN)
            .filter(self._gen_conds(self.filter_attributes))
            .withColumn("max_total_generation", f.max("total_generation").over(w))
            .withColumn("max_SER_dinamic", f.max("SER_dinamic").over(w))
        )
        return data_sp

    def get_training_data(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ):
        data_sp = self._get_basic_portfolios(spark, start_date, end_date, params)
        data_sp = data_sp.filter(f.col("drop_flg") == 0)

        #         data = spark_df_toPandas(
        #             data_sp,
        #             write_cache=False,
        #             read_cache=False,
        #             )

        weird_flag = MetaDataLoader._gen_weird_flag_spark(data_sp)

        # фильтруем только правильные наблюдения
        weird_flag = weird_flag.filter(weird_flag.weird_flag != 1)

        # джойним слева только корректные наблюдения
        data = weird_flag.join(data_sp, on=["gen_name"], how="left")

        data = data.toPandas()

        return {"data": data}

    def get_prediction_data(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, DataFrame]:
        return {"features": DataFrame()}

    def get_ground_truth(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, DataFrame]:
        data_sp = self._get_basic_portfolios(spark, start_date, end_date, params)

        data = spark_df_toPandas(
            data_sp,
            write_cache=False,
            read_cache=False,
        )

        data.loc[:, "weird_flag"] = MetaDataLoader._gen_weird_flag(data)
        data.loc[:, "newbiz_flag"] = data.groupby("gen_name")["open_month"].transform(
            "min"
        ) >= start_date.strftime(self.date_format)

        return {"target": data}


class MetaTrainer(Meta):
    def __new__(cls, model_params, abc_base):
        attrs = dict()
        attrs["model_type"] = "Trainer"
        attrs["class_name"] = gen_class_name(CONFIG, model_params, attrs["model_type"])
        attrs["file_name"] = gen_file_name(CONFIG, model_params)
        attrs["date_format"] = "%Y-%m"

        attrs["dataloader"] = MetaDataLoader(model_params, DataLoader)()

        attrs.update(model_params)
        attrs.update(CONFIG["common_params"])

        return super().__new__(cls, attrs["class_name"], abc_base, attrs)

    def _get_training_data(
        self, spark, end_date: datetime, start_date: datetime
    ) -> None:
        self.training_data: Dict[str, DataFrame] = self.dataloader.get_training_data(
            spark, start_date, end_date
        )

    def get_trained_model(
        self,
        spark,
        end_date: datetime,
        start_date: datetime = None,
        hyperparams: Dict[str, Any] = None,
    ):
        if start_date is None:
            start_date = self.start_date
        if hyperparams is None:
            hyperparams = self.hyperparams

        self._get_training_data(spark, end_date, start_date)

        model = self.model_class(**self.model_class_kwargs)

        return model.fit(self.training_data["data"], **hyperparams)

    def save_trained_model(
        self,
        spark,
        saving_path: str,
        end_date: datetime,
        start_date: datetime = None,
        overwrite: bool = True,
        hyperparams: Dict[str, Any] = None,
    ) -> bool:
        if start_date is None:
            start_date = self.start_date
        if hyperparams is None:
            hyperparams = self.hyperparams

        fname = (
            f"{self.file_name}_{dt_convert(start_date)}_{dt_convert(end_date)}.pickle"
        )

        if overwrite == False:
            if os.path.isfile(os.path.join(saving_path, fname)):
                pickle_name = fname
                return pickle_name
        try:
            file = open(os.path.join(saving_path, fname), "wb")
            model_fit = self.get_trained_model(spark, end_date, start_date, hyperparams)
            dump(model_fit, file)
            output = fname
        except (OSError, PickleError, RecursionError) as e:
            print(e)
            output = None

        return output


class MetaAdapter(Meta):
    def __new__(cls, model_params, abc_base):
        attrs = dict()
        attrs["model_type"] = "Adapter"
        attrs["class_name"] = gen_class_name(CONFIG, model_params, attrs["model_type"])
        attrs["file_name"] = gen_file_name(CONFIG, model_params)
        attrs["date_format"] = "%Y-%m"
        attrs.update(model_params)
        attrs.update(CONFIG["common_params"])

        return super().__new__(cls, attrs["class_name"], abc_base, attrs)

    def _gen_pd_filter(self, attribute: str, attr_params: List[str]):
        # attr_params: List[str] = getattr(self, attribute)
        attr_map: Dict[str, Dict[str, List[str]]] = getattr(self, attribute + "_map")
        conds = []
        for param_value in attr_params:
            col_value_map = attr_map[param_value]
            # cond = reduce(lambda x, y: (x & y), [f.col(col).isin(val) for col, val in col_value_map.items()])
            cond_expr = " and ".join(
                [f"({col} in {val})" for col, val in col_value_map.items()]
            )
            conds.append(cond_expr)
        return "(" + ") or (".join(conds) + ")"

    def _gen_query_expr(self, attributes):
        filters = [
            self._gen_pd_filter(attribute, getattr(self, attribute))
            for attribute in attributes
        ]

        return "(" + ") and (".join(filters) + ")"

    def _basic_filter_portfolio(self, portfolio: DataFrame) -> DataFrame:
        # This filters needed opt, segment, or other attrs in portfolio_filter_attrs
        query_expr = self._gen_query_expr(self.portfolio_filter_attrs)
        return portfolio.query(query_expr)

    def _split_normal_weird(self, portfolio: DataFrame) -> Tuple[DataFrame, DataFrame]:
        # splits incoming portfolio into two:
        # 1) normal portfolio which can be forecasted with the model
        # 2) weird portfolio for which no redemption is assumed

        weird_flag = MetaDataLoader._gen_weird_flag(portfolio)

        return (
            portfolio[~weird_flag].assign(weird_flag=False).copy(),
            portfolio[weird_flag].assign(weird_flag=True).copy(),
        )

    def _port_max_rates(self, dataframe):
        # this method calculates max opening rate for the existing portfolio
        # (basically a copy of _gen_external_max_rate_scenario but for different col names)

        rates_list = [f"report_weight_open_rate_{i}m" for i in self.periods_list]

        if len(rates_list) > 1:
            return dataframe[rates_list].max(axis=1)
        else:
            return dataframe[rates_list[0]]

    def _gen_external_filter(self, external_config):
        filters = [
            self._gen_pd_filter(attribute, external_config[attribute])
            for attribute in self.portfolio_filter_attrs
        ]
        return "(" + ") and (".join(filters) + ")"

    def _gen_external_max_rate_scenario(self, first_pred_date):
        scenario_rates = self.scenario_data.loc[
            self.scenario_data.index >= first_pred_date, :
        ][self.scenario_vtb_rates_cols]

        return scenario_rates.max(axis=1).to_frame().rename(columns={0: "max_rate"})

    def _gen_rate_scenario(self, pred_date, period, freq="M"):
        if freq == "M":
            period = find_left_nearest_point(period, MONTH_MATURITY_MAP_)

            scenario_rate = self.scenario_data.loc[
                pred_date,
                get_feature_name(
                    "VTB_weighted_rate",
                    self.segment[0],
                    self.replenishable_flg,
                    self.subtraction_flg,
                    MONTH_MATURITY_MAP_[period],
                ),
            ]
        elif freq == "D":
            period = find_left_nearest_point(period, MATURITY_)
            scenario_rate = self.scenario_data.loc[
                pred_date,
                get_feature_name(
                    "VTB_weighted_rate",
                    self.segment[0],
                    self.replenishable_flg,
                    self.subtraction_flg,
                    period,
                ),
            ]
        else:
            raise ValueError(
                f'{freq} value for freq is not supported. Try one of ["M", "D"]'
            )

        return scenario_rate

    def _gen_inflow_predicted(self, pred_date, period, freq="M"):
        if freq == "M":
            period = find_left_nearest_point(period, MONTH_MATURITY_MAP_)

            inflow = self.model_data[self.features_key].loc[
                pred_date,
                get_feature_name(
                    "y_inflow",
                    self.segment[0],
                    self.replenishable_flg,
                    self.subtraction_flg,
                    MONTH_MATURITY_MAP_[period],
                ),
            ]
        elif freq == "D":
            period = find_left_nearest_point(period, MATURITY_)
            inflow = self.model_data[self.features_key].loc[
                pred_date,
                get_feature_name(
                    "y_inflow",
                    self.segment[0],
                    self.replenishable_flg,
                    self.subtraction_flg,
                    period,
                ),
            ]
        else:
            raise ValueError(
                f'{freq} value for freq is not supported. Try one of ["M", "D"]'
            )

        return inflow

    def _gen_external_weighted_rate(self, first_pred_date):
        weighted_rates = self.model_data[self.features_key].loc[
            :, self.weighted_vtb_rate
        ]

        return weighted_rates[weighted_rates.index >= first_pred_date].to_frame()

    def _make_prediction(self, port):
        # this method actually calculates predictions of SER_dinamic_cl

        port_mod, exog_vars = early_redemption_ptr_transform(
            port, self.cols_to_split, self.labels, self.tresholds, self.features
        )

        # print(port[port_mod[exog_vars+['gen_name', _REPORT_DT_COLUMN]].isna()])
        # print(port_mod[exog_vars+['gen_name', _REPORT_DT_COLUMN]].isna().sum())
        assert (
            port_mod[exog_vars + ["gen_name", _REPORT_DT_COLUMN]].isna().sum().sum()
            == 0
        ), "NA values found in port_mod. Aborting due to resulting misssing forecast"

        assert (
            self.reg.params.index.to_list() == exog_vars
        ), "Different features in reg and exog_vars"

        preds = self.reg.predict(
            port_mod.set_index(["gen_name", _REPORT_DT_COLUMN])[exog_vars]
        ).reset_index(drop=False)
        preds.loc[:, "predictions"] = np.where(
            (preds.predictions > 0) & (port.optional_flg.isin([0, 1])),
            0,
            np.where(preds.predictions < -1, -1, preds.predictions),
        )
        # unfortunately self.reg.predict returns a df of differnet shape, if there is None in exogs
        # we usually have none in exogs if the generetaion is new and has no "spread_weight_rate_&_weight_open_rate_lag"

        return preds

    def _single_external_evolution(self, portfolio, first_pred_date, weird_flag):
        # this method iteratively constructs prediction-ready dataframe and
        # passes it into _make_prediction method
        # then it calculates total_generation_cleared, total_generation for this iteration

        max_rates = self._gen_external_max_rate_scenario(first_pred_date)
        weighted_rates = self._gen_external_weighted_rate(first_pred_date)

        # last_weight_rate = (
        #     portfolio
        #     .sort_values(['gen_name', _REPORT_DT_COLUMN])
        #     .groupby('gen_name')
        #     ['weight_rate'].last()
        # )

        res = [
            portfolio.sort_values([_REPORT_DT_COLUMN, "gen_name"]).reset_index(
                drop=True
            )
        ]
        for pred_date in [d for d in self.forecast_dates if d >= first_pred_date]:
            f_port = pd.DataFrame().reindex_like(res[-1])

            f_port.loc[:, _REPORT_DT_COLUMN] = pred_date
            f_port.loc[:, "report_month"] = f_port.loc[
                :, _REPORT_DT_COLUMN
            ].dt.strftime(self.date_format)
            const_cols = [
                # эти колонки реально должны быть константными при прогнозе на 1 месяц
                "gen_name",
                "open_month",
                "close_month",
                "is_vip_or_prv",
                "drop_flg",
                "optional_flg",
                "weird_flag",
                "bucketed_period",
                "bucketed_open_rate",
                "bucketed_balance",
                "CUR",
                "init_total_generation",
                # по сути взвешенная ставка должна меняться
                # тк после реализованного досрочного отзыва может измениться взвешенная ставка поколения
                # но учитывая, что в данном поколении содержатся очень близкие вклады по ставкам - не критично
                "weight_rate",
                "weight_renewal_cnt",
                "weight_renewal_available_flg",
                "weight_close_plan_day",
            ]

            for col in const_cols:
                f_port.loc[:, col] = res[-1][col]

            f_port.sort_values([_REPORT_DT_COLUMN, "gen_name"], inplace=True)
            f_port["close_month"] = np.where(
                f_port.close_month > "2050-01", "2050-01", f_port.close_month
            )

            # re-calculate cols that change with time
            f_port.loc[:, "share_period_plan"] = (
                f_port[_REPORT_DT_COLUMN] - pd.to_datetime(f_port["open_month"])
            ).dt.days / (
                pd.to_datetime(f_port["close_month"])
                - pd.to_datetime(f_port["open_month"])
            ).dt.days

            f_port.loc[:, "spread_weight_rate_&_weight_open_rate"] = (
                res[-1].loc[:, "weight_rate"] - max_rates.loc[pred_date, "max_rate"]
            )

            f_port.loc[:, "months_left"] = res[-1].loc[:, "months_left"] - 1
            f_port.loc[:, "total_generation_cl_lag1"] = res[-1][
                "total_generation_cleared"
            ]
            f_port.loc[:, "total_generation_lag1"] = res[-1]["total_generation"]
            f_port.loc[:, "report_wo_period_weight_open_rate"] = weighted_rates.loc[
                pred_date, self.weighted_vtb_rate
            ]
            f_port.loc[:, "spread_weight_rates_wo_period"] = (
                f_port.loc[:, "weight_rate"]
                - f_port.loc[:, "report_wo_period_weight_open_rate"]
            )

            f_port.loc[f_port["optional_flg"].isin([1, 3]), "incentive"] = f_port.loc[
                f_port["optional_flg"].isin([1, 3]), "spread_weight_rates_wo_period"
            ]

            # Для вкладов без опции снятия
            f_port.loc[f_port["optional_flg"].isin([0, 2]), "incentive"] = f_port.loc[
                f_port["optional_flg"].isin([0, 2]),
                "spread_weight_rate_&_weight_open_rate",
            ]

            f_port.loc[:, "incentive_lag1"] = res[-1]["incentive"]
            f_port.loc[:, "incentive_lag1"] = f_port.loc[:, "incentive_lag1"].fillna(0)

            # now calculate predictions of SER_dinamic_cl
            if weird_flag or self.reg is None:
                f_port.loc[:, "SER_dinamic_cl"] = 0
            else:
                prediction = self._make_prediction(f_port)
                prediction.sort_values([_REPORT_DT_COLUMN, "gen_name"], inplace=True)
                f_port.loc[:, "SER_dinamic_cl"] = prediction["predictions"].values

            # находим очищенный от процентов объем поколения
            f_port.loc[:, "total_generation_cleared"] = f_port[
                "total_generation_cl_lag1"
            ] * (1 + f_port["SER_dinamic_cl"])
            f_port.loc[:, "SER_d_cl"] = (
                f_port["total_generation_cleared"] - res[-1]["total_generation_cleared"]
            )

            # тут начинается очень тонкая материя: думаем о поколении как об одном вкладе
            # происходит попытка учесть то, что, как правило, проценты начисляются на минимальный остаток за месяц
            # если у нас досрочка положительная (были пополнения) - считаем процентный доход от баланса в прошлый месяц
            # если досрочка отрицательная (в рамках пополнения она воспринимается как расходные операции по сути)
            # то считаем проценты на современный момент (с учетом расходов)
            # Везде считается, что проценты капитализируются и выплачиваются раз в месяц
            interest_income = np.where(
                f_port.loc[:, "SER_dinamic_cl"] >= 0,
                f_port["total_generation_lag1"] * f_port["weight_rate"] / 12 / 100,
                (f_port["total_generation_lag1"] + f_port["SER_d_cl"])
                * f_port["weight_rate"]
                / 12
                / 100,
            )

            f_port.loc[:, "total_interests"] = (
                res[-1]["total_interests"] + interest_income
            )
            f_port.loc[:, "remaining_interests"] = (
                res[-1]["remaining_interests"] - interest_income
            )
            f_port.loc[:, "total_generation"] = (
                f_port["total_generation_lag1"] + f_port["SER_d_cl"] + interest_income
            )
            f_port.loc[:, "SER_d"] = (
                f_port["total_generation"] - f_port["total_generation_lag1"]
            )
            f_port.loc[:, "SER_dinamic"] = f_port["SER_d"] / (
                f_port["total_generation_lag1"] + 1
            )
            for period in self.periods_list:
                f_port.loc[
                    :, f"report_weight_open_rate_{period}m"
                ] = self._gen_rate_scenario(pred_date, period)

            f_port.loc[:, "row_count"] = np.where(
                res[-1]["row_count"] > 0, res[-1]["row_count"] + 1, -1
            )
            f_port.loc[:, "max_total_generation"] = np.maximum(
                res[-1]["max_total_generation"].fillna(0), f_port["total_generation"]
            )
            f_port.loc[:, "max_SER_dinamic"] = np.maximum(
                res[-1]["max_SER_dinamic"].fillna(0), f_port["SER_dinamic"]
            )
            f_port = f_port.query("report_month < close_month")
            f_port = f_port.sort_values([_REPORT_DT_COLUMN, "gen_name"]).reset_index(
                drop=True
            )
            res.append(f_port)

        return pd.concat(res[1:])[PORTFOLIO_COLUMNS_]

    def _portfolio_evolution(self, portfolio: DataFrame, weird_flag) -> DataFrame:
        # берет ставки из сценария для нового бизнеса
        #

        # Рассчитываем показатели, нужные для модели, но на дату портфеля (на прошлый месяц).
        # По сути все это в модель не пойдет, но это нужно для расчета показателей на дату прогноза
        # Например, портфель тут лежит на 2022-12-31. Мы хотим предиктить в 2023-01-31.
        # Важная фича в модели - incentive_lag1. Чтобы знать эту фичу на 2023-01-31 мы должны рассчитать incentive на 2022-12-31
        # incentive, который рассчитывается тут будет той самой лаговой фичей в след месяце.
        portfolio.loc[:, "max_rate"] = self._port_max_rates(portfolio)
        portfolio.loc[:, "spread_weight_rate_&_weight_open_rate"] = (
            portfolio.loc[:, "weight_rate"] - portfolio.loc[:, "max_rate"]
        )

        portfolio.loc[:, "spread_weight_rates_wo_period"] = (
            portfolio.loc[:, "weight_rate"]
            - portfolio.loc[:, "report_wo_period_weight_open_rate"]
        )

        portfolio.loc[:, "months_passed"] = portfolio["row_count"] - 1
        portfolio.loc[:, "months_left"] = (
            portfolio["bucketed_period"] + 1 - portfolio["months_passed"]
        )

        portfolio.loc[
            portfolio["optional_flg"].isin([1, 3]), "incentive"
        ] = portfolio.loc[
            portfolio["optional_flg"].isin([1, 3]), "spread_weight_rates_wo_period"
        ]

        # Для вкладов без опции снятия
        portfolio.loc[
            portfolio["optional_flg"].isin([0, 2]), "incentive"
        ] = portfolio.loc[
            portfolio["optional_flg"].isin([0, 2]),
            "spread_weight_rate_&_weight_open_rate",
        ]
        portfolio.loc[:, "incentive_lag1"] = portfolio.groupby("gen_name")[
            "incentive"
        ].shift()

        ew_result = self._single_external_evolution(
            portfolio, self.forecast_dates[0], weird_flag
        )
        return ew_result

    def _create_renewal_generations(self, portfolio):
        renewal_port = portfolio.query("renewal_balance_next_month > 0")
        first_pred_date = self.forecast_dates[0]
        weighted_rates = self._gen_external_weighted_rate(first_pred_date)

        res = [
            renewal_port.sort_values([_REPORT_DT_COLUMN, "gen_name"]).reset_index(
                drop=True
            )
        ]
        for pred_date in [d for d in self.forecast_dates if d >= first_pred_date]:
            f_port = pd.DataFrame().reindex_like(res[-1])
            f_port.loc[:, _REPORT_DT_COLUMN] = pred_date
            f_port.loc[:, "report_month"] = f_port.loc[
                :, _REPORT_DT_COLUMN
            ].dt.strftime(self.date_format)

            f_port.loc[:, "open_month"] = f_port["report_month"]

            const_cols = [
                # по подходу Вовы в досрочке пролонгированные вклады получают значение
                # drop_flg = 1 (или другое, если есть помимо этого другие косяки)
                "bucketed_balance",
                "is_vip_or_prv",
                "optional_flg",
                "bucketed_period",
                "CUR",
                # будем считать, что если вклад пролонгировался один раз, то может и второй
                # что конечно же не всегда верно (но это вероятно не слишком критично для короткого прогноза)
                # поэтому weight_renewal_available_flg = const
                # также константа взвешенная дата закрытия, тк обычно срочность - целое число месяцев
                "weight_renewal_available_flg",
                "weight_close_plan_day",
            ]

            for col in const_cols:
                f_port.loc[:, col] = res[-1][col]

            zero_cols = [
                "total_interests",
                "SER_d",
                "SER_d_cl",
                "SER_dinamic",
                "SER_dinamic_cl",
                "share_period_plan",
                "max_SER_dinamic",
            ]
            f_port[zero_cols] = 0.0

            f_port.loc[:, "total_generation"] = res[-1]["renewal_balance_next_month"]

            # drop_flg - закодированная проблема вклада (типо 10101) где каждая позиция означает за тип проблемы
            # если 1 - проблема есть, 0 - проблемы нет
            # если мы видим в разряде единиц значение 1 - значит поколение пролонгировано
            # drop_flg при этом может принимать значения 1, 11, 101, 111, 1001, 1101 и тд,
            f_port.loc[:, "drop_flg"] = np.where(
                res[-1]["drop_flg"] % 2 == 1,
                res[-1]["drop_flg"],
                res[-1]["drop_flg"] + 1,
            )
            f_port.loc[:, "drop_flg"] = f_port.loc[:, "drop_flg"].fillna(1)

            def change_close_month(close_month, bucketed_period, default_period=12):
                if (
                    (not bucketed_period)
                    or (bucketed_period < 2)
                    or (np.isnan(bucketed_period))
                ):
                    bucketed_period = default_period
                close_date = pd.to_datetime(close_month) + pd.DateOffset(
                    months=bucketed_period - 1
                )
                close_month = close_date.strftime("%Y-%m")
                return close_month

            # эта штука долговато работает (а progress_apply недоступен в текущей версии пандаса)
            f_port.loc[:, "close_month"] = res[-1].apply(
                lambda row: change_close_month(
                    row["close_month"], row["bucketed_period"]
                ),
                axis=1,
            )
            f_port.loc[:, "weight_rate"] = np.where(
                res[-1]["weight_renewal_available_flg"] > 0.01,
                res[-1].bucketed_period.apply(
                    lambda x: self._gen_rate_scenario(pred_date, x - 1)
                ),
                0.01,
            )
            f_port.loc[:, "bucketed_open_rate"] = f_port["weight_rate"].apply(
                lambda x: bisect_left(self.rate_buckets, x)
            )
            f_port.loc[:, "remaining_interests"] = (
                f_port["total_generation"]
                * (f_port["weight_rate"] / 12 / 100)
                * (f_port["bucketed_period"] - 1)
            )
            f_port.loc[:, "total_generation_cleared"] = f_port["total_generation"]
            f_port.loc[:, "weight_renewal_cnt"] = res[-1]["weight_renewal_cnt"] + 1
            f_port.loc[:, "total_generation_lag1"] = None
            f_port.loc[:, "total_generation_cl_lag1"] = None

            gen_name_cols = [
                "report_month",
                "close_month",
                "bucketed_balance",
                "bucketed_period",
                "bucketed_open_rate",
                "optional_flg",
                "is_vip_or_prv",
                "drop_flg",
                "CUR",
            ]

            for col in gen_name_cols:
                f_port["gen_name"] = reduce(
                    lambda x, y: x + "_" + y.astype(str),
                    [f_port[col] for col in gen_name_cols],
                )

            for period in self.periods_list:
                f_port.loc[
                    :, f"report_weight_open_rate_{period}m"
                ] = self._gen_rate_scenario(pred_date, period)
            f_port.loc[:, "report_wo_period_weight_open_rate"] = weighted_rates.loc[
                pred_date, self.weighted_vtb_rate
            ]
            f_port.loc[:, "init_total_generation"] = f_port["total_generation"]
            f_port.loc[:, "row_count"] = 1
            f_port.loc[:, "max_total_generation"] = f_port["total_generation"]

            res.append(f_port)
        return pd.concat(res[1:])[PORTFOLIO_COLUMNS_]

    def get_bucket_balance_share(self, forecast_date, portfolio, window):
        """Расчитываем историческое распределение по бакетам объема при открытии за последние window месяцев"""
        start_date = (forecast_date + MonthEnd(-window)).strftime("%Y-%m")
        port = portfolio[portfolio.open_month >= start_date]
        bucket_balance_share = (
            port.groupby("bucketed_balance")["init_total_generation"].sum()
            / port["init_total_generation"].sum()
        ).reset_index()
        bucket_balance_share = bucket_balance_share.rename(
            columns={"init_total_generation": "total_generation_share"}
        )

        normal_buckets = pd.DataFrame({"bucketed_balance": self.balance_buckets})
        # может быть так, что люди в определенном бакете объема просто не открывали вкладов за последние window месяцев, поэтому
        # нужно причесать результат под ожидаемые бакеты
        res = normal_buckets.merge(
            bucket_balance_share, on="bucketed_balance", how="left"
        ).fillna(0)
        # если вообще не было открытий за последние window месяцев в конкретном сегменте и опциональности - сумма будет равна нулю
        # тк bucket_balance_share будет пустой таблицей. Тогда считаем, что равномерно распределены (конечно же неправда)
        if res.total_generation_share.sum() == 0:
            res.total_generation_share = 1 / res.shape[0]
        if abs(res.total_generation_share.sum() - 1) > 0.01:
            raise ValueError("Bucket distribution sum is not equal to 1")
        return res

    def create_newbusiness_generations(self, curr_portfolio):
        first_pred_date = self.forecast_dates[0]
        weighted_rates = self._gen_external_weighted_rate(first_pred_date)
        bucket_balance_share = self.get_bucket_balance_share(
            first_pred_date, curr_portfolio, window=6
        )
        segment = self.segment_map[self.segment[0]]["is_vip_or_prv"][0]
        optionality = self.optionality_map[self.optionality[0]]["optional_flg"][0]
        currency = self.cur[0]
        res = []
        for pred_date in [d for d in self.forecast_dates if d >= first_pred_date]:
            f_port = pd.DataFrame()
            for maturity in MATURITY_:
                # тут заполняются колонки, которые отличаются для разных срочностей
                newbiz_maturity = pd.DataFrame(
                    columns=PORTFOLIO_COLUMNS_, index=bucket_balance_share.index
                )
                newbiz_maturity.loc[:, "bucketed_balance"] = bucket_balance_share[
                    "bucketed_balance"
                ]
                newbiz_maturity.loc[:, "total_generation"] = bucket_balance_share[
                    "total_generation_share"
                ] * self._gen_inflow_predicted(pred_date, period=maturity, freq="D")
                newbiz_maturity.loc[:, "bucketed_period"] = (
                    MATURITY_TO_MONTH_MAP_[maturity] + 1
                )
                newbiz_maturity.loc[:, "close_month"] = (
                    pred_date + pd.DateOffset(months=MATURITY_TO_MONTH_MAP_[maturity])
                ).strftime("%Y-%m")
                newbiz_maturity.loc[:, "weight_rate"] = self._gen_rate_scenario(
                    pred_date, maturity, freq="D"
                )
                f_port = f_port.append(newbiz_maturity)

            # тут заполняются колонки, которые являются общими для разных срочностей
            f_port.loc[:, _REPORT_DT_COLUMN] = pred_date
            f_port.loc[:, "report_month"] = f_port.loc[
                :, _REPORT_DT_COLUMN
            ].dt.strftime(self.date_format)
            f_port.loc[:, "open_month"] = f_port["report_month"]
            f_port.loc[:, "is_vip_or_prv"] = segment
            f_port.loc[:, "optional_flg"] = optionality
            f_port.loc[:, "drop_flg"] = 0
            f_port.loc[:, "bucketed_open_rate"] = f_port["weight_rate"].apply(
                lambda x: bisect_left(self.rate_buckets, x)
            )
            f_port.loc[:, "remaining_interests"] = (
                f_port["total_generation"]
                * (f_port["weight_rate"] / 12 / 100)
                * (f_port["bucketed_period"] - 1)
            )
            f_port.loc[:, "total_generation_cleared"] = f_port["total_generation"]
            f_port.loc[:, "total_generation_lag1"] = None
            f_port.loc[:, "total_generation_cl_lag1"] = None
            f_port.loc[:, "weight_renewal_available_flg"] = 1
            f_port.loc[:, "weight_close_plan_day"] = 15.0
            f_port.loc[:, "CUR"] = currency

            gen_name_cols = [
                "report_month",
                "close_month",
                "bucketed_balance",
                "bucketed_period",
                "bucketed_open_rate",
                "optional_flg",
                "is_vip_or_prv",
                "drop_flg",
                "CUR",
            ]
            for col in gen_name_cols:
                f_port["gen_name"] = reduce(
                    lambda x, y: x + "_" + y.astype(str),
                    [f_port[col] for col in gen_name_cols],
                )

            for period in self.periods_list:
                f_port.loc[
                    :, f"report_weight_open_rate_{period}m"
                ] = self._gen_rate_scenario(pred_date, period)
            zero_cols = [
                "weight_renewal_cnt",
                "total_interests",
                "SER_d",
                "SER_d_cl",
                "SER_dinamic",
                "SER_dinamic_cl",
                "share_period_plan",
                "max_SER_dinamic",
            ]
            f_port[zero_cols] = 0

            f_port.loc[:, "report_wo_period_weight_open_rate"] = weighted_rates.loc[
                pred_date, self.weighted_vtb_rate
            ]
            f_port.loc[:, "init_total_generation"] = f_port["total_generation"]
            f_port.loc[:, "row_count"] = 1
            f_port.loc[:, "max_total_generation"] = f_port["total_generation"]

            f_port = f_port[f_port.total_generation > 0].reset_index(drop=True)
            res.append(f_port)
        return pd.concat(res, ignore_index=True)[PORTFOLIO_COLUMNS_]

    def predict(
        self,
        forecast_context: ForecastContext,
        portfolio: pd.DataFrame = None,
        **params,
    ) -> Any:
        self.forecast_dates = forecast_context.forecast_dates
        self.scenario_data: pd.DataFrame = forecast_context.scenario.scenario_data

        self.model_data = forecast_context.model_data
        curr_portfolio = forecast_context.model_data[self.portfolio_key][
            forecast_context.portfolio_dt
        ]

        curr_portfolio = self._basic_filter_portfolio(curr_portfolio)
        if curr_portfolio.shape[0] == 0:
            return curr_portfolio
        curr_normal_port, curr_weird_port = self._split_normal_weird(curr_portfolio)

        self.reg = self._model_meta[0]
        self.target = self._model_meta[1]
        self.features = self._model_meta[2]

        (
            self.cols_to_split,
            self.labels,
            self.tresholds,
        ) = self.model_class.PTR_input_transform(
            self._model_meta[3], self._model_meta[4], self._model_meta[5]
        )

        curr_portfolio_pred_normal = self._portfolio_evolution(
            curr_normal_port, weird_flag=False
        )
        curr_portfolio_pred_weird = self._portfolio_evolution(
            curr_weird_port, weird_flag=True
        )

        ew_curr_portfolio_pred = (
            pd.concat(
                (
                    curr_portfolio_pred_normal,
                    curr_portfolio_pred_weird,
                )
            )
            .query("total_generation >= 0")
            .reset_index(drop=True)
        )

        renewal_port_pred = self._create_renewal_generations(curr_portfolio)
        newbusiness_pred = self.create_newbusiness_generations(curr_portfolio)

        portfolio_pred = pd.concat(
            [newbusiness_pred, ew_curr_portfolio_pred, renewal_port_pred]
        ).reset_index(drop=True)

        return portfolio_pred

    def _make_prediction_in_sample(self, portfolio: pd.DataFrame):
        """
        Этот метод создан для получения прогноза модели на тренировочных данных (in sample).
        Предполагается для аналитики работы моделей в классaх типа CalculatorAnalyzer.
        """

        portfolio = portfolio.copy()

        portfolio.loc[:, "max_rate"] = self._port_max_rates(portfolio)

        portfolio.loc[:, "spread_weight_rate_&_weight_open_rate"] = (
            portfolio.loc[:, "weight_rate"] - portfolio.loc[:, "max_rate"]
        )

        portfolio.loc[:, "spread_weight_rates_wo_period"] = (
            portfolio.loc[:, "weight_rate"]
            - portfolio.loc[:, "report_wo_period_weight_open_rate"]
        )

        portfolio.loc[:, "months_passed"] = portfolio["row_count"] - 1
        portfolio.loc[:, "months_left"] = (
            portfolio["bucketed_period"] + 1 - portfolio["months_passed"]
        )

        portfolio.loc[
            portfolio["optional_flg"].isin([1, 3]), "incentive"
        ] = portfolio.loc[
            portfolio["optional_flg"].isin([1, 3]), "spread_weight_rates_wo_period"
        ]

        # Для вкладов без опции снятия
        portfolio.loc[
            portfolio["optional_flg"].isin([0, 2]), "incentive"
        ] = portfolio.loc[
            portfolio["optional_flg"].isin([0, 2]),
            "spread_weight_rate_&_weight_open_rate",
        ]
        portfolio.loc[:, "incentive_lag1"] = portfolio.groupby("gen_name")[
            "incentive"
        ].shift()

        # Дроп Na значений для корректной работы модели (они могут появится в ходе преобразований метода)
        portfolio = portfolio.dropna(subset=["incentive", "incentive_lag1"])

        port_mod, exog_vars = early_redemption_ptr_transform(
            portfolio, self.cols_to_split, self.labels, self.tresholds, self.features
        )

        preds = self.reg.predict(
            port_mod.set_index(["gen_name", _REPORT_DT_COLUMN])[exog_vars]
        ).reset_index(drop=False)

        preds.loc[:, "predictions"] = (
            preds.loc[:, "predictions"]
            .mask(
                (preds.predictions > 0) & port_mod.optional_flg.isin([0, 1]),
                0,  # Для модели для вкладов без пополнений предикт <=0
            )
            .mask(
                (preds.predictions < -1),
                -1,  # Доля отозванных вкладов не может быть больше -1
            )
        )

        return preds


# class MetaAdapter1(Meta):
#     def __new__(cls, model_params, abc_base):
#         attrs = dict()
#         attrs['model_type'] = "Adapter"
#         attrs['class_name'] = gen_class_name(CONFIG, model_params, attrs['model_type'])
#         attrs['file_name'] = gen_file_name(CONFIG, model_params)
#         attrs['date_format'] = "%Y-%m"
#         attrs.update(model_params)
#         attrs.update(CONFIG['common_params'])

#         return super().__new__(cls, attrs['class_name'], abc_base, attrs)

#     def _create_vintage_df(self, forecast_context):
#         dates = forecast_context.forecast_dates
#         terms = forecast_context.model_data['war_mass_noopt'].columns.drop([_REPORT_DT_COLUMN])
#         df = pd.DataFrame()
#         for opt in ['opt', 'noopt']:
#             sizes = list(self.size_structure[self.size_structure.segment.apply(lambda x: x.split('_')[1]) == opt]['size'].unique())
#             segments = [f'vip_{opt}', f'mass_{opt}']
#             df_opt = pd.DataFrame(list(product(dates, terms, segments, sizes)), columns = [_REPORT_DT_COLUMN, 'term', 'segment', 'size'])
#             df = df.append(df_opt, ignore_index = True)
#         return df

#     def _gen_pd_filter(self, attribute: str, attr_params: List[str]):
#         #attr_params: List[str] = getattr(self, attribute)
#         attr_map: Dict[str, Dict[str, List[str]]] = getattr(self, attribute+"_map")
#         conds = []
#         for param_value in attr_params:
#             col_value_map = attr_map[param_value]
#             #cond = reduce(lambda x, y: (x & y), [f.col(col).isin(val) for col, val in col_value_map.items()])
#             cond_expr = " and ".join([f"({col} in {val})" for col, val in col_value_map.items()])
#             conds.append(cond_expr)
#         return "(" + ") or (".join(conds) + ")"

#     def _gen_query_expr(self, attributes):

#         filters = [
#             self._gen_pd_filter(attribute, getattr(self, attribute))
#             for attribute in attributes
#         ]

#         return "(" + ") and (".join(filters) + ")"

#     def _basic_filter_portfolio(self, portfolio: DataFrame) -> DataFrame:
#         # This filters needed opt, segment, or other attrs in portfolio_filter_attrs
#         query_expr = self._gen_query_expr(self.portfolio_filter_attrs)
#         return portfolio.query(query_expr)

#     def _split_normal_weird(self, portfolio: DataFrame
#                             ) -> Tuple[DataFrame, DataFrame]:
#         # splits incoming portfolio into two:
#         # 1) normal portfolio which can be forecasted with the model
#         # 2) weird portfolio for which no redemption is assumed

#         weird_flag = MetaDataLoader._gen_weird_flag(portfolio)

#         return (
#             portfolio[~weird_flag].assign(weird_flag=False).copy(),
#             portfolio[weird_flag].assign(weird_flag=True).copy()
#         )

#     def _generate_newbiz_portfolio(self, external_config, fdate, empty_df) -> DataFrame:
#         # generates newbiz generations for given date, external_config,
#         # and data from other models
#         # empty_df is needed mainly for format purposes

#         scenario_vtb_rates_cols = [
#             f"{external_config['scenario_vtb_rates_prefix']}.{m}"
#             for m in external_config['maturity_buckets']
#         ]
#         scenario_rates = (
#             self.scenario_data
#             .set_index(_REPORT_DT_COLUMN)
#             .loc[fdate, scenario_vtb_rates_cols]
#         )
#         maturity_structure_cols = [
#             f"{external_config['maturity_structure_col_prefix']}.{m}"
#             for m in external_config['maturity_buckets']
#         ]
#         maturity_structure = (
#             self.model_data
#             [external_config['maturity_structure_first_model_key']]
#             [external_config['maturity_structure_second_model_key']]
#             .loc[fdate, maturity_structure_cols]
#         )
#         size_structure_cols = [
#             f"{external_config['size_structure_col_prefix']}.{m}"
#             for m in external_config['size_buckets']
#         ]
#         size_structure = (
#             self.model_data
#             [external_config['size_structure_first_model_key']]
#             [external_config['size_structure_second_model_key']]
#             .loc[fdate, size_structure_cols]
#         )
#         newbiz_volume = (
#             self.model_data
#             [external_config['newbusiness_first_model_key']]
#             [external_config['newbusiness_second_model_key']]
#             .loc[fdate, external_config['newbusiness_col_name']]
#         )

#         size_len = len(external_config['size_buckets'])
#         mat_len = len(external_config['maturity_buckets'])

#         for size_index, mat_index in product(range(size_len), range(mat_len)):
#             rate = scenario_rates.iloc[mat_index]
#             maturity_share = maturity_structure.iloc[mat_index]
#             maturity_int = external_config['maturity_buckets_int'][mat_index]
#             size_share = size_structure.iloc[size_index]
#             size_int = external_config['size_buckets_int'][size_index]

#             init_balance = newbiz_volume * size_share * maturity_share
#             if init_balance > 0:
#                 pass
#             else:
#                 # это вклады со сроком 25+ месяцев
#                 continue
#             mean_balance = init_balance / (size_int * 1000)
#             count_agreements = init_balance / mean_balance

#             report_month = fdate.strftime(self.date_format)
#             close_month = (fdate + relativedelta(months=maturity_int)).strftime(self.date_format)
#             maturity_days = ((fdate + relativedelta(months=maturity_int)) - fdate).days

#             remaining_interests = init_balance * (1 + rate/100)**(maturity_days/365) - init_balance

#             vip_flg = int(''.join([str(i) for i in getattr(self, "segment_map")[external_config['segment'][0]]['vip_flg']]))
#             opt_flg = int(''.join([str(i) for i in getattr(self, "optionality_map")[external_config['optionality'][0]]['optional_flg']]))
#             cur = ''.join([str(i) for i in getattr(self, "cur_map")[external_config['cur'][0]]['cur']])

#             size_buckets = [0, 100, 400, 1_000, 2_000, 1e12]
#             mat_buckets = [30*n for n in range(87)]
#             rate_buckets = list(range(100))

#             bucketed_balance = float(bisect_left(size_buckets, size_int))
#             bucketed_period = float(bisect_left(mat_buckets, maturity_int))
#             bucketed_open_rate = float(bisect_left(rate_buckets, rate))

#             gen_data = {
#                 _REPORT_DT_COLUMN: fdate,
#                 'report_month': report_month,
#                 'open_month': report_month,
#                 'close_month': close_month,
#                 'vip_flg': vip_flg,
#                 'drop_flg': 0,
#                 'optional_flg': opt_flg,
#                 'bucketed_balance': bisect_left(size_buckets, size_int),
#                 'bucketed_period': bisect_left(mat_buckets, maturity_int),
#                 'bucketed_open_rate': bisect_left(rate_buckets, rate),
#                 'avg_contract_period': maturity_days,
#                 'avg_rate': rate,
#                 'weight_rate': rate,
#                 'early_close_generation': 0.,
#                 'early_close_generation_cl': 0.,
#                 'avg_init_balance_amt': mean_balance,
#                 'avg_end_balance_amt': None,
#                 'total_interests': 0.,
#                 'remaining_interests': remaining_interests,
#                 'total_generation': init_balance,
#                 'total_generation_cleared': init_balance,
#                 'count_agreements': count_agreements,
#                 'SER_d': 0.,
#                 'SER_d_cl': 0.,
#                 'SER': 0.,
#                 'SER_cl': 0.,
#                 'SER_dinamic': 0.,
#                 'SER_dinamic_cl': 0.,
#                 'total_generation_lag1': None,
#                 'total_generation_cl_lag1': None,
#                 'cur': cur,
#                 'share_period': 0.,
#                 'days_passed': 0,
#                 'days_plan': maturity_days,
#                 'share_period_plan': 0.,
#                 'opt_flg_binary': int(opt_flg == 0),
#                 'months_passed': 0,
#                 'months_left': bucketed_period + 1

#             }


#             # 'report_perc_10', 'report_mean_open_rate', 'report_weight_open_rate',
#             # , 'row_count',
#             # 'max_total_generation', 'max_SER_dinamic']

#             gen_name = f"{report_month}_{close_month}_{bucketed_balance}_{bucketed_period}_{bucketed_open_rate}_{opt_flg}_{vip_flg}_0_{cur}"
#             gen_data['gen_name'] = gen_name

#             empty_df = empty_df.append(gen_data, ignore_index=True)

#         mat_map = {'1m': 0, '3m': 0, '6m': 1, '12m': 2, '24m': 4}
#         for k, v in mat_map.items():
#             empty_df[f'report_weight_open_rate_{k}'] = scenario_rates[f"{external_config['scenario_vtb_rates_prefix']}.{external_config['maturity_buckets'][v]}"]

#         empty_df['weird_flag'] = False
#         empty_df['newbiz_flag'] = True
#         empty_df['init_total_generation'] = empty_df['total_generation']
#         empty_df['init_count_agreements'] = empty_df['count_agreements']
#         empty_df['init_weight_rate'] = empty_df['weight_rate']
#         empty_df['row_count'] = 1
#         # print(empty_df.shape)
#         # print(empty_df.gen_name.value_counts())
#         return empty_df.reset_index(drop=True)

#     def _gen_external_filter(self, external_config):
#         filters = [
#             self._gen_pd_filter(attribute, external_config[attribute])
#             for attribute in self.portfolio_filter_attrs
#         ]
#         return "(" + ") and (".join(filters) + ")"

#     def _split_portfolio_to_externals(self, portfolio: DataFrame):

#         ports = []
#         # print(portfolio.groupby([_REPORT_DT_COLUMN, 'vip_flg'])['total_generation', 'total_generation_cleared'].mean())


#         for external_config in self.external_configs:
#             query = self._gen_external_filter(external_config)
#             # print(query)
#             ports.append(
#                 (
#                     external_config,
#                     portfolio.query(query)
#                 )
#             )
#             # print(portfolio.query(query).groupby([_REPORT_DT_COLUMN, 'vip_flg'])['total_generation', 'total_generation_cleared'].mean())

#         return ports

#     def _gen_external_max_rate_scenario(self, external_config, first_pred_date):

#         scenario_vtb_rates_cols = [
#             f"{external_config['scenario_vtb_rates_prefix']}.{m}"
#             for m in external_config['maturity_buckets']]

#         scenario_rates = (
#             self.scenario_data
#             .query(f"{_REPORT_DT_COLUMN}>='{first_pred_date}'")
#             .set_index(_REPORT_DT_COLUMN)
#             [scenario_vtb_rates_cols]
#         )

#         return scenario_rates.max(axis=1).to_frame().rename(columns={0: 'max_rate'})#.reset_index(drop=False).rename(columns={0: 'max_rate'})

#     def _port_max_rates(self, dataframe):
#         # this method calculates max opening rate for the existing portfolio
#         # (basically a copy of _gen_external_max_rate_scenario but for different col names)
#         periods_list = [24, 12, 6, 3, 1]

#         rates_list = [f'report_weight_open_rate_{i}m' for i in periods_list]

#         if len(rates_list) > 1:
#             return dataframe[rates_list].max(axis=1)
#         else:
#             return dataframe[rates_list[0]]

#     def _make_prediction(self, port):
#         # this method actually calculates predictions of SER_dinamic_cl

#         port_mod, exog_vars = early_redemption_ptr_transform(
#             port,
#             self.cols_to_split,
#             self.labels,
#             self.tresholds,
#             self.features
#         )

#         # print(port[port_mod[exog_vars+['gen_name', _REPORT_DT_COLUMN]].isna()])
#         # print(port_mod[exog_vars+['gen_name', _REPORT_DT_COLUMN]].isna().sum())
#         assert port_mod[exog_vars+['gen_name', _REPORT_DT_COLUMN]].isna().sum().sum() == 0, (
#             'NA values found in port_mod. Aborting due to resulting misssing forecast'
#         )

#         preds = (
#             self.reg
#             .predict(
#                 port_mod
#                 .set_index(['gen_name', _REPORT_DT_COLUMN])
#                 [exog_vars]
#             )
#             .reset_index(drop=False)
#         )
#         preds.loc[:, 'predictions'] = np.where(
#             (preds.predictions > 0) & (port.optional_flg.isin([0,1]) ),
#             0,
#             preds.predictions
#         )
#         # unfortunately self.reg.predict returns a df of differnet shape, if there is None in exogs
#         # we usually have none in exogs if the generetaion is new and has no "spread_weight_rate_&_weight_open_rate_lag"

#         return preds

#     def _single_external_evolution(self, external_config, portfolio, first_pred_date, weird_flag):
#         # this method iteratively constructs prediction-ready dataframe and
#         # passes it into _make_prediction method
#         # then it calculates total_generation_cleared, total_generation for this iteration

#         #print(1)
#         #print(portfolio.groupby([_REPORT_DT_COLUMN, 'vip_flg'])['total_generation', 'total_generation_cleared'].sum())

#         max_rates = self._gen_external_max_rate_scenario(
#             external_config,
#             first_pred_date
#         )

#         last_weight_rate = (
#             portfolio
#             .sort_values(['gen_name', _REPORT_DT_COLUMN])
#             .groupby('gen_name')
#             ['weight_rate'].last()
#         )


#         res = [portfolio.sort_values([_REPORT_DT_COLUMN, 'gen_name']).reset_index(drop=True)]
#         #print(res[0].columns)
#         for pred_date in [d for d in self.forecast_dates if d>=first_pred_date]:


#             #print(3, pred_date)
#             #print(res[-1].groupby([_REPORT_DT_COLUMN, 'vip_flg'])['total_generation', 'total_generation_cleared'].sum())

#             # copy format without data
#             f_port = pd.DataFrame().reindex_like(res[-1])

#             # set appropriate report dt and report_month
#             f_port.loc[:, _REPORT_DT_COLUMN] = pred_date
#             f_port.loc[:, 'report_month'] = f_port.loc[:, _REPORT_DT_COLUMN].dt.strftime(self.date_format)

#             # these cols don't change with time
#             const_cols = [
#                 # эти колонки реально должны быть константными при прогнозе на 1 месяц
#                 'gen_name', 'open_month', 'close_month',
#                 'vip_flg', 'drop_flg', 'optional_flg', 'weird_flag',
#                 'bucketed_period', 'bucketed_open_rate', 'bucketed_balance',

#                 # эти в теории нет, но опустим этот момент
#                 # по сути эти колонки нужно пересчитать на после прогноза досрочки на нужную дату
#                 # тк после реализованного досрочного отзыва может измениться взвешенная ставка поколения
#                 # но учитывая, что в данном поколении содержатся очень близкие вклады по ставкам - не критично
#                 # count_agreements конечно же точно изменится, но это поле при прогнозе по сути не нужно
#                 # (возможно удалю его после окончательного понимания, что оно не нужно)
#                 # поля в нижней строке точно нужно пересчитывать уже после прогноза
#                 # тк если мы прогнозируем пополнения, то max_total_generation может увеличиться
#                 # аналогично для max_SER_dinamic
#                 'weight_rate', 'count_agreements',
#                 'max_total_generation', 'max_SER_dinamic',
#             ]
#             for col in const_cols:
#                 f_port.loc[:, col] = res[-1][col]

#             f_port.sort_values([_REPORT_DT_COLUMN, 'gen_name'], inplace=True)

#             # re-calculate cols that change with time
#             f_port.loc[:, 'share_period_plan'] = (
#                 (
#                     f_port[_REPORT_DT_COLUMN] - pd.to_datetime(f_port['open_month'])
#                 ).dt.days
#                 / (
#                     pd.to_datetime(f_port['close_month']) - pd.to_datetime(f_port['open_month'])
#                 ).dt.days
#             )

#             # this will set spread_weight_rate_&_weight_open_rate_lag1 to None for new generations
#             # проверить что тут имеется в виду: лаг спреда или лаг ставок открытия
#             f_port.loc[:, 'spread_weight_rate_&_weight_open_rate_lag1'] = (
#                 res[-1]['spread_weight_rate_&_weight_open_rate']
#             )

#             f_port.loc[:, 'spread_weight_rate_&_weight_open_rate'] = (
#                 res[-1].loc[:, 'weight_rate'] - max_rates.loc[pred_date, 'max_rate']
#             )

#             # hence we need to fill it with something (for example zero)
#             f_port.loc[:, 'spread_weight_rate_&_weight_open_rate_lag1'] = (
#                 f_port.loc[:, 'spread_weight_rate_&_weight_open_rate_lag1'].fillna(0)
#             )


#             f_port.loc[:, 'months_left'] = res[-1].loc[:, 'months_left'] - 1
#             f_port.loc[:, 'total_generation_cl_lag1'] = res[-1]['total_generation_cleared']
#             f_port.loc[:, 'total_generation_lag1'] = res[-1]['total_generation']

#             # для безопц немного другая фича:
#             f_port.loc[:, 'incentive'] = (res[-1].loc[:, 'weight_rate'] * (f_port['bucketed_period'] + 1) - f_port['months_left'] * max_rates.loc[pred_date, 'max_rate']) / 12
#             # объелиним все вместе на основе флага опциональности (от 0 до 3 вкл.)
#             f_port.loc[:, 'incentive'] = (
#                 ((f_port['optional_flg']==0).astype(float) * f_port['incentive'] ).values
#                 + ((f_port['optional_flg']!=0).astype(float) * f_port['spread_weight_rate_&_weight_open_rate']).values
#             )
#             f_port.loc[:, 'incentive_lag1'] = res[-1]['incentive']
#             f_port.loc[:, 'incentive_lag1'] = f_port.loc[:, 'incentive_lag1'].fillna(0)
#             #print(f_port.loc[:, 'incentive_lag1'].isna().sum())

#             # now calculate predictions of SER_dinamic_cl
#             if weird_flag:
#                 f_port.loc[:, 'SER_dinamic_cl'] = 0
#             else:
#                 prediction = self._make_prediction(f_port)
#                 prediction.sort_values([_REPORT_DT_COLUMN, 'gen_name'], inplace=True)
#                 f_port.loc[:, 'SER_dinamic_cl'] = prediction['predictions'].values

#             # add derived columns:
#             # here we assume that percentages are paid before the early withdrowal
#             interest_income = f_port['total_generation_lag1'] * f_port['weight_rate'] / 12 / 100

#             f_port.loc[:, 'total_interests'] = res[-1]['total_interests'] + interest_income
#             f_port.loc[:, 'remaining_interests'] = res[-1]['remaining_interests'] - interest_income
#             f_port.loc[:, 'total_generation'] = (
#                 (res[-1]['total_generation'] + interest_income)
#                 *
#                 (1 + f_port['SER_dinamic_cl'])
#             )
#             f_port.loc[:, 'SER_d'] = f_port['total_generation'] - res[-1]['total_generation']
#             f_port.loc[:, 'total_generation_cleared'] = f_port['total_generation_cl_lag1'] * (1 + f_port['SER_dinamic_cl'])
#             f_port.loc[:, 'SER_d_cl'] = f_port['total_generation_cleared'] - res[-1]['total_generation_cleared']
#             # print(f_port[['total_interests', 'remaining_interests', 'total_generation', 'early_close_generation', 'total_generation_cleared']].isna().sum())
#             # print(f_port.groupby([_REPORT_DT_COLUMN, 'vip_flg'])[['total_interests', 'remaining_interests', 'total_generation', 'early_close_generation', 'total_generation_cleared']].sum())
#             res.append(f_port)

#         return pd.concat(res).query('report_month <= close_month')

#     def _portfolio_evolution(self, portfolio: DataFrame, weird_flag) -> DataFrame:
#         # берет ставки из сценария для нового бизнеса
#         #
#         #print('before')
#         #print(portfolio.groupby([_REPORT_DT_COLUMN, 'vip_flg'])['total_generation', 'total_generation_cleared'].sum())
#         portfolio.loc[:, 'max_rate'] = self._port_max_rates(portfolio)
#         portfolio.loc[:, 'spread_weight_rate_&_weight_open_rate'] = (
#                 portfolio.loc[:, 'weight_rate'] - portfolio.loc[:, 'max_rate']
#         )

#         portfolio.loc[:, 'months_passed'] = portfolio['row_count'] - 1
#         portfolio.loc[:, 'months_left'] = portfolio['bucketed_period'] + 1 - portfolio['months_passed']

#         portfolio.loc[:,'incentive'] = (portfolio['weight_rate'] * (portfolio['bucketed_period'] + 1) - portfolio['months_left'] * portfolio.loc[:, 'max_rate']) / 12
#         # объелиним все вместе на основе флага опциональности (от 0 до 3 вкл.)
#         portfolio.loc[:,'incentive'] = (portfolio['optional_flg']==0)*portfolio['incentive'] + (portfolio['optional_flg']!=0)*portfolio['spread_weight_rate_&_weight_open_rate']
#         portfolio.loc[:,'incentive_lag1'] = portfolio.groupby('gen_name')['incentive'].shift()
#         #print(portfolio[['incentive', 'incentive_lag1']])
#         #print('after')
#         #print(portfolio.groupby([_REPORT_DT_COLUMN, 'vip_flg'])['total_generation', 'total_generation_cleared'].sum())
#         ports = self._split_portfolio_to_externals(portfolio)

#         current_portfolios = []
#         newbiz_portfolios = []
#         for external_config, ext_port in ports:

#             cur = ''.join([str(i) for i in getattr(self, "cur_map")[external_config['cur'][0]]['cur']])
#             #print(ext_port.groupby([_REPORT_DT_COLUMN, 'vip_flg'])['total_generation', 'total_generation_cleared'].sum())
#             current_portfolios.append(
#                 self._single_external_evolution(
#                     external_config,
#                     ext_port,
#                     self.forecast_dates[0],
#                     weird_flag
#                 )
#                 .assign(cur=cur)
#             )

#             if weird_flag==False:
#                 # no weird portfolios in new business
#                 for i, fdate in enumerate(self.forecast_dates):
#                     # generate & evaluate newbiz for every forecast date
#                     if fdate == self.forecast_dates[-1]:
#                         # no need to predict for the last date new portfolio
#                         newbiz_portfolios.append(
#                             self._generate_newbiz_portfolio(
#                                     external_config,
#                                     fdate,
#                                     pd.DataFrame(columns=ext_port.columns)
#                                 )
#                         )
#                     else:
#                         newbiz_portfolios.append(
#                             self._single_external_evolution(
#                                 external_config,
#                                 self._generate_newbiz_portfolio(
#                                     external_config,
#                                     fdate,
#                                     pd.DataFrame(columns=ext_port.columns)
#                                 ),
#                                 self.forecast_dates[i+1],
#                                 weird_flag
#                             )
#                         )
#             else:
#                 newbiz_portfolios.append(pd.DataFrame())

#         return pd.concat(current_portfolios).reset_index(drop=True), pd.concat(newbiz_portfolios).reset_index(drop=True)

#     def _portfolio_aggregator(self, gen_portfolio: DataFrame) -> DataFrame:
#         # aggregates over generations
#         return (
#             gen_portfolio
#             .query('report_month < close_month')
#             # .groupby([_REPORT_DT_COLUMN, 'vip_flg'], as_index=False)
#             .groupby([_REPORT_DT_COLUMN, 'weird_flag'], as_index=False)
#             [[
#                 'total_generation', 'total_generation_cleared',
#                 'total_interests', 'remaining_interests',
#                 'SER_d', 'SER_d_cl'
#             ]]
#             .agg('sum')#[['total_generation', 'total_generation_cleared']]
#         )

#     def predict(self, forecast_context: ForecastContext, portfolio: pd.DataFrame = None, **params) -> Any:

#         self.forecast_dates = forecast_context.forecast_dates
#         #print(self.forecast_dates)
#         self.scenario_data: pd.DataFrame = forecast_context.scenario.scenario_data

#         # TODO: перенести forecast_context.model_data в аргумент для
#         # _portfolio_evolution и _generate_newbiz_portfolio, чтобы не копировать model_data в адаптер
#         self.model_data = forecast_context.model_data

#         curr_portfolio = forecast_context.model_data[self.portfolio_key]


#         curr_portfolio = self._basic_filter_portfolio(curr_portfolio)
#         curr_normal_port, curr_weird_port = self._split_normal_weird(curr_portfolio)

#         self.reg = self._model_meta[0]
#         self.target = self._model_meta[1]
#         self.features = self._model_meta[2]


#         self.cols_to_split, self.labels, self.tresholds = self.model_class.PTR_input_transform(
#             self._model_meta[3],
#             self._model_meta[4],
#             self._model_meta[5]
#             )

#         # gen_name is like
#         # 2010-01_2019-01_36.0_0.0_2_0_1111_EUR
#         curr_portfolio_pred_normal, newbiz_portfolio_pred = self._portfolio_evolution(curr_normal_port, weird_flag=False)
#         curr_portfolio_pred_weird, _ = self._portfolio_evolution(curr_weird_port, weird_flag=True)

#         curr_portfolio_pred = pd.concat(
#             (
#                 curr_portfolio_pred_normal,
#                 curr_portfolio_pred_weird,
#             )
#         )
#         curr_portfolio_pred['newbiz_flag'] = False

#         return {
#             'current_portfolio': curr_portfolio_pred,
#             'current_portfolio_agg': self._portfolio_aggregator(curr_portfolio_pred),
#             'newbiz_portfolio': newbiz_portfolio_pred,
#             'newbiz_portfolio_agg': self._portfolio_aggregator(newbiz_portfolio_pred),
#         }

# немного уличной магии для возможности пиклинга динамических классов
# gloabals() создает классы здесь, причем с правильным названием, а в meta.py они просто импортятся
for model_params in CONFIG["models_params"]:
    loader_name = gen_class_name(CONFIG, model_params, "DataLoader")
    globals()[loader_name] = MetaDataLoader(model_params, DataLoader)

    trainer_name = gen_class_name(CONFIG, model_params, "Trainer")
    globals()[trainer_name] = MetaTrainer(model_params, ModelTrainer)

    adapter_name = gen_class_name(CONFIG, model_params, "Adapter")
    globals()[adapter_name] = MetaAdapter(model_params, BaseModel)


if __name__ == "__main__":
    for model_params in CONFIG["models_params"]:
        print(gen_class_name(CONFIG, model_params, "DataLoader"))
        print(gen_file_name(CONFIG, model_params))

# if __name__=='__main__':

#     from calendar import monthrange
#     import pickle

#     from dateutil.relativedelta import relativedelta
#     from upfm.commons import ForecastContext, ModelInfo, Scenario


#     model_file = '/home/vtb4044606/projects/deposit-early-redemption/test/deposit_earlyredemption_noopt_vip_novip_RUR_201401_202001.pickle'

#     adapter_class = MetaAdapter(CONFIG['models_params'][0], BaseModel)
#     info = ModelInfo.from_str(model_file.split('/')[-1].split('.')[0])
#     adapter = adapter_class(info, model_file)
#     print(adapter)

#     dt_ = datetime(year=2020, month=1, day=31)
#     horizon_ = 3
#     multiplier = 1
#     horizon_ = horizon_ * multiplier

#     forecast_dates_ = [(dt_ + relativedelta(months=m+1)) for m in range(horizon_)]
#     forecast_dates_ = [dt.replace(day=monthrange(dt.year, dt.month)[1]) for dt in forecast_dates_]
#     scenario_data = {
#         # 'report_dt': forecast_dates_,
#         'RUB.DEPOSITS.NOVIP.NOOPT.VTB.RATE.1-3m': [0.07105, 0.05105, 0.05105],
#         'RUB.DEPOSITS.NOVIP.NOOPT.VTB.RATE.4-6m': [0.05220, 0.05220, 0.05220],
#         'RUB.DEPOSITS.NOVIP.NOOPT.VTB.RATE.7-12m': [0.05420, 0.05420, 0.05420],
#         'RUB.DEPOSITS.NOVIP.NOOPT.VTB.RATE.13-18m': [0.05560, 0.05560, 0.05560],
#         'RUB.DEPOSITS.NOVIP.NOOPT.VTB.RATE.19-24m': [0.05560, 0.05560, 0.05560],
#         'RUB.DEPOSITS.NOVIP.NOOPT.VTB.RATE.25-36m': [0.05740, 0.05740, 0.05740],
#         'RUB.DEPOSITS.NOVIP.NOOPT.VTB.RATE.36+m': [0.051100, 0.051100, 0.051100],

#         'RUB.DEPOSITS.VIP.NOOPT.VTB.RATE.1-3m': [0.05105, 0.05105, 0.05105],
#         'RUB.DEPOSITS.VIP.NOOPT.VTB.RATE.4-6m': [0.05220, 0.05220, 0.05220],
#         'RUB.DEPOSITS.VIP.NOOPT.VTB.RATE.7-12m': [0.05420, 0.05420, 0.05420],
#         'RUB.DEPOSITS.VIP.NOOPT.VTB.RATE.13-18m': [0.05560, 0.05560, 0.05560],
#         'RUB.DEPOSITS.VIP.NOOPT.VTB.RATE.19-24m': [0.05560, 0.05560, 0.05560],
#         'RUB.DEPOSITS.VIP.NOOPT.VTB.RATE.25-36m': [0.05740, 0.05740, 0.05740],
#         'RUB.DEPOSITS.VIP.NOOPT.VTB.RATE.36+m': [0.051100, 0.051100, 0.051100],

#         'RUB.DEPOSITS.NOVIP.OPT.VTB.RATE.1-3m': [0.05105, 0.05105, 0.05105],
#         'RUB.DEPOSITS.NOVIP.OPT.VTB.RATE.4-6m': [0.05220, 0.05220, 0.05220],
#         'RUB.DEPOSITS.NOVIP.OPT.VTB.RATE.7-12m': [0.05420, 0.05420, 0.05420],
#         'RUB.DEPOSITS.NOVIP.OPT.VTB.RATE.13-18m': [0.05560, 0.05560, 0.05560],
#         'RUB.DEPOSITS.NOVIP.OPT.VTB.RATE.19-24m': [0.05560, 0.05560, 0.05560],
#         'RUB.DEPOSITS.NOVIP.OPT.VTB.RATE.25-36m': [0.05740, 0.05740, 0.05740],
#         'RUB.DEPOSITS.NOVIP.OPT.VTB.RATE.36+m': [0.051100, 0.051100, 0.051100],

#         'RUB.DEPOSITS.VIP.OPT.VTB.RATE.1-3m': [0.05105, 0.05105, 0.05105],
#         'RUB.DEPOSITS.VIP.OPT.VTB.RATE.4-6m': [0.05220, 0.05220, 0.05220],
#         'RUB.DEPOSITS.VIP.OPT.VTB.RATE.7-12m': [0.05420, 0.05420, 0.05420],
#         'RUB.DEPOSITS.VIP.OPT.VTB.RATE.13-18m': [0.05560, 0.05560, 0.05560],
#         'RUB.DEPOSITS.VIP.OPT.VTB.RATE.19-24m': [0.05560, 0.05560, 0.05560],
#         'RUB.DEPOSITS.VIP.OPT.VTB.RATE.25-36m': [0.05740, 0.05740, 0.05740],
#         'RUB.DEPOSITS.VIP.OPT.VTB.RATE.36+m': [0.051100, 0.051100, 0.051100],

#         'RUB.DEPOSITS.RUONIA': [0.065, 0.06, 0.065],
#         'RUB.DEPOSIT.SBER.SHARE': [0.5, 0.5, 0.5],
#         'RUB.SA.VTB.RATE': [0.07, 0.07, 0.07],
#         'RUB.OFZ.1Y':[0.08,0.08,0.08],
#         'RUB.DEPOSIT.OUTFLOW.PLAN':[10**9,10**9,10**9]
#     }

#     scenario_data = {k: [val*100 for val in v] * multiplier for k, v in scenario_data.items()}
#     scenario_data['report_dt'] = forecast_dates_

#     print(forecast_dates_)

#     scenario_data = pd.DataFrame.from_dict({**scenario_data})
#     # portfolio_dt: datetime, horizon: int, scenario_data: DataFrame
#     scenario_ = Scenario(
#         portfolio_dt = dt_,
#         horizon=horizon_,
#         scenario_data=scenario_data,
#         )

#     port = pd.read_pickle('/home/vtb4044606/projects/deposit-early-redemption/test/port_202001.pickle')

#     model_data = pickle.load(open('/home/vtb4044606/projects/deposit-early-redemption/test/model_data.pickle', 'rb'))
#     model_data[CONFIG['common_params']['portfolio_key']] = port

#     #print(model_data.shape)
#     context = ForecastContext(
#         dt_,
#         forecast_horizon=horizon_,
#         scenario=scenario_,
#         model_data=model_data
#     )

#     preds = adapter.predict(context)

#     print(preds)
