import os.path
import numpy as np
import pandas as pd
import pyspark.sql.functions as f

from pandas import DataFrame
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, Any, Tuple, List
from pickle import dump, load, PickleError

from upfm.commons import (
    ModelTrainer,
    DataLoader,
    BaseModel,
    ModelInfo,
    ForecastContext,
)

from deposit_early_redemption.spark_pandas import spark_df_toPandas
from deposit_early_redemption.base_model import EarlyRedemptionNoOptModel


def dt_convert(dt: datetime, delimiter: str = "") -> str:
    return delimiter.join(str(dt.date()).split("-")[:2])


_DEFAULT_START_DATE_NOOPT = datetime(2014, 1, 31)
_DEFAULT_HYPERPARAMS_NOOPT = dict(
    n_iter=2,
    p_step=20,
    p_low=10,
    p_high=90,
    add_constant=False,
    scaler=False,
    entity_effects=True,
    lasso_alpha=None,
    rank_check_in_iters=False,
)

_SEGMENTS = ["novip_noopt", "novip_opt", "vip_noopt", "vip_opt"]


class EarlyRedemptionNooptTrainer(ModelTrainer):
    def __init__(self):
        self.default_hyperparams: Dict[str, Any] = _DEFAULT_HYPERPARAMS_NOOPT
        self.current_hyperparams: Dict[str, Any] = _DEFAULT_HYPERPARAMS_NOOPT

        self.default_start_date: datetime = _DEFAULT_START_DATE_NOOPT
        self.current_start_date: datetime = _DEFAULT_START_DATE_NOOPT

    def _get_training_data(
        self, spark, end_date: datetime, start_date: datetime
    ) -> None:
        self.dataloader: EarlyRedemptionNooptDataLoader = (
            EarlyRedemptionNooptDataLoader()
        )
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
            start_date = self.default_start_date
        if hyperparams is None:
            hyperparams = self.default_hyperparams

        self._get_training_data(spark, end_date, start_date)

        model = EarlyRedemptionNoOptModel("RUR", 1)

        return model.fit(self.training_data["data"], **hyperparams)

    def save_trained_model(
        self,
        spark,
        saving_path: str,
        end_date: datetime,
        start_date: datetime = None,
        hyperparams: Dict[str, Any] = None,
    ) -> bool:
        if start_date is None:
            start_date = self.default_start_date
        if hyperparams is None:
            hyperparams = self.default_hyperparams

        fname = f"deposits_earlyredemption_regr_rurnoopt_{dt_convert(start_date)}_{dt_convert(end_date)}.pickle"

        try:
            file = open(os.path.join(saving_path, fname), "wb")
            model_fit = self.get_trained_model(spark, end_date, start_date, hyperparams)
            dump(model_fit, file)
            output = fname
        except (OSError, PickleError, RecursionError) as e:
            print(e)
            output = None

        return output


class EarlyRedemptionNooptDataLoader(DataLoader):
    def __init__(self):
        self._DEFAULT_START_DATE = _DEFAULT_START_DATE_NOOPT
        self.date_format = "%Y-%m"

    def _date_extractor(self, string: str) -> datetime:
        init = datetime.strptime(string, self.date_format)
        return init + relativedelta(months=1) - timedelta(days=1)

    def get_maximum_train_range(self, spark) -> Tuple[datetime, datetime]:
        a = spark.sql(
            """
                      select min(open_month) as `min`,
                      max(open_month) as `max`
                      from dadm_alm_sbx.deposits_early_close_v0
                      """
        ).collect()

        min_open_month = self._date_extractor(a[0].asDict()["min"])
        max_open_month = self._date_extractor(a[0].asDict()["max"])

        return max(min_open_month, self._DEFAULT_START_DATE), max_open_month

    def _max_rates(self, dataframe: DataFrame, periods_list: List[int]) -> DataFrame:
        rates_list = [f"report_weight_open_rate_{i}m" for i in periods_list]
        if len(rates_list) > 1:
            return np.where(
                dataframe["months_left"] >= periods_list[0],
                dataframe[rates_list].max(axis=1),
                self._max_rates(dataframe, periods_list[1:]),
            )
        else:
            return dataframe[rates_list[0]]

    def _portfolio_data_transform(
        self, dat: DataFrame, periods_list: List[int]
    ) -> DataFrame:
        dat["open_dt"] = pd.to_datetime(dat.open_month) + pd.offsets.MonthEnd(1)
        dat["close_dt"] = pd.to_datetime(dat.close_month) + pd.offsets.MonthEnd(1)

        for col in dat.columns:
            try:
                dat[col] = dat[col].astype(np.float64)
            except Exception as e:
                print(e)
                pass

        dat["months_passed"] = dat["row_count"] - 1
        dat["months_left"] = dat["bucketed_period"] + 1 - dat["months_passed"]

        periods_list = sorted(periods_list, reverse=True)

        dat["spread_weight_rate_&_weight_open_rate"] = dat[
            "weight_rate"
        ] - self._max_rates(dat, periods_list)

        dat["SER_dinamic_cl_lag1"] = dat.groupby("gen_name")["SER_dinamic_cl"].shift()

        for col in dat.columns:
            if "spread" in col:
                dat[f"{col}_lag1"] = dat.groupby("gen_name")[col].shift()

        return dat

    def _get_current_portfolio(
        self, spark, first_predict_date: datetime
    ) -> Dict[str, DataFrame]:
        periods_list = [24, 12, 6, 3, 1]

        date_t1 = first_predict_date - relativedelta(months=1)
        date_t2 = first_predict_date - 2 * relativedelta(months=1)

        sql = f"""
            select * from dadm_alm_sbx.deposits_early_close_v0
            where 1=1
                and CUR = "RUR"
                and optional_flg = 0
                and close_month > report_month 
                and report_month <= '{date_t1.strftime(self.date_format)}' 
                and report_month >= '{date_t2.strftime(self.date_format)}'
            """
        # print(sql)
        # raise ValueError()
        # data = spark.sql(sql).toPandas()
        data = spark_df_toPandas(
            spark.sql(sql), write_cache=True, read_cache=True, convert_decimals=True
        )

        data = self._portfolio_data_transform(data, periods_list)

        data = data.query(f"report_month == '{dt_convert(date_t1, '-')}'")[
            [
                "gen_name",
                "report_date",
                "open_dt",
                "close_dt",
                "bucketed_period",
                "spread_weight_rate_&_weight_open_rate",
                "spread_weight_rate_&_weight_open_rate_lag1",
                "share_period_plan",
                "months_left",
                "total_generation_cleared",
                "weight_rate",
                "CUR",
                "vip_share",
                "drop_flg",
            ]
            + [f"report_weight_open_rate_{m}m" for m in periods_list]
        ]

        return data

    def _training_data_transform_sp(self, spark_df):
        # drop generations with max balance <= 100k for RUR
        dat1 = spark_df.groupby("gen_name").agg(
            f.max("total_generation").alias("max_total_generation")
        )
        spark_df = spark_df.join(dat1, "gen_name", "left")
        spark_df = spark_df.where(f.col("max_total_generation") > 10**5).drop(
            "max_total_generation"
        )
        spark_df = spark_df.toDF(*spark_df.columns)

        # drop generations with max SER_dinamic > 5
        dat1 = spark_df.groupby("gen_name").agg(
            f.max("SER_dinamic").alias("max_SER_dinamic")
        )
        spark_df = spark_df.join(dat1, "gen_name", "left")
        spark_df = spark_df.where(f.col("max_SER_dinamic") <= 5).drop("max_SER_dinamic")
        spark_df = spark_df.toDF(*spark_df.columns)

        return spark_df

    def _training_data_transform_pd(self, pd_df):
        pd_df = pd_df.assign(
            open_dt=pd.to_datetime(pd_df.open_month) + pd.offsets.MonthEnd(1)
        ).assign(close_dt=pd.to_datetime(pd_df.close_month) + pd.offsets.MonthEnd(1))
        pd_df = pd_df.assign(
            strange_flg=(
                True
                & (pd_df["SER_dinamic"] < -0.05)
                & (
                    (pd_df["close_dt"] - pd_df["report_date"]).apply(lambda x: x.days)
                    < 33
                )
                & (
                    pd_df["report_weight_open_rate_24m"]
                    > pd_df["report_weight_open_rate_3m"]
                )
            )
        )

        pd_df = pd_df.assign(
            strange_flg=pd_df.groupby("gen_name")["strange_flg"].transform("max")
        )
        pd_df = pd_df[~pd_df["strange_flg"]]
        return pd_df

    def get_training_data(self, spark, start_date, end_date):
        sql = f"""
        select * from dadm_alm_sbx.deposits_early_close_v0
        where 1=1
            and CUR = "RUR"
            and optional_flg = 0
            and close_month > report_month
            and drop_flg = 0
            and report_month <= '{end_date.strftime(self.date_format)}' 
            and report_month >= '{start_date.strftime(self.date_format)}'
        """
        data_sp = spark.sql(sql)
        data_sp = self._training_data_transform_sp(data_sp)

        data = spark_df_toPandas(
            data_sp,
            write_cache=True,
            read_cache=True,
            convert_decimals=True,
            infer_method=None,
        )

        data = self._training_data_transform_pd(data)

        return {"data": data}

    def get_prediction_data(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, DataFrame]:
        freq = params.get("freq", "M") if params else "M"

        if freq == "M":
            ruonia: DataFrame = (
                self._get_ruonia(spark, start_date, end_date)
                .resample("M")
                .last()
                .reset_index()
            )
            ruonia = ruonia.rename(columns={"rate": "RUONIA_smooth"})
        elif freq == "D":
            ruonia: DataFrame = self._get_ruonia(spark, start_date, end_date)
        else:
            raise ValueError(
                "Invalid value of param 'freq'. Only 'M' and 'D' are supported"
            )

        return {"features": ruonia}

    def get_ground_truth(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, DataFrame]:
        freq = params.get("freq", "M") if params else "M"

        if freq == "M":
            target: DataFrame = (
                self._get_target(spark, start_date, end_date)
                .resample("M")
                .last()
                .reset_index()
            )
        elif freq == "D":
            target: DataFrame = self._get_target(spark, start_date, end_date)
        else:
            raise ValueError(
                "Invalid value of param 'freq'. Only 'M' and 'D' are supported"
            )

        return {"target": target}


class EarlyRedemptionModelAdapter(BaseModel):
    def __init__(self, model_info_: ModelInfo, file_path: str) -> None:
        super().__init__(model_info_)
        self._file_path = file_path
        with open(self._file_path, "rb") as in_file:
            self._model_meta = load(in_file)

        # [reg, self.target, self.features, m_cols, m_labels, m_tresh]
        self.reg = self._model[0]
        self.target = self._model[1]
        self.features = self._model[2]

        (
            self.cols_to_split,
            self.labels,
            self.tresholds,
        ) = EarlyRedemptionNoOptModel.PTR_input_transform(
            self._model[3], self._model[4], self._model[5]
        )

    def predict(
        self,
        forecast_context: ForecastContext,
        portfolio: pd.DataFrame = None,
        **params,
    ) -> Any:
        forecast_dates = forecast_context.forecast_dates
        scenario_data: pd.DataFrame = forecast_context.scenario.scenario_data
        model_data = forecast_context.model_data

        # gen_name is like
        # 2010-01_2019-01_36.0_0.0_2_0_1111_EUR

        # pred_future.index.name = _REPORT_DT_COLUMN

        portfolio = 1

        params: np.array = np.atleast_2d(np.asarray(self.params)).T
        predictions: np.array = portfolio.values @ params

        return predictions

    def _create_vintage_df(self, forecast_context):
        dates = forecast_context.forecast_dates
        terms = forecast_context.model_data["war_mass_noopt"].columns.drop(
            ["report_dt"]
        )
        df = pd.DataFrame()
        for opt in ["opt", "noopt"]:
            sizes = list(
                self.size_structure[
                    self.size_structure.segment.apply(lambda x: x.split("_")[1]) == opt
                ]["size"].unique()
            )
            segments = [f"vip_{opt}", f"mass_{opt}"]
            df_opt = pd.DataFrame(
                list(product(dates, terms, segments, sizes)),
                columns=["report_dt", "term", "segment", "size"],
            )
            df = df.append(df_opt, ignore_index=True)
        return df

    def generate_newbiz_portfolio(self) -> DataFrame:
        pass

    def portfolio_evolution(self, portfolio: DataFrame) -> DataFrame:
        pass

    @property
    def model(self) -> Any:
        return self._model_meta


class EarlyRedemptionModelAdapter_old(BaseModel):
    def __init__(self, model_info: ModelInfo, file_path) -> None:
        super().__init__(model_info)
        self._file_path = file_path
        with open(self._file_path, "rb") as infile:
            self._model = pickle.load(infile)
            self.params = self._model[0]
            self.target = self._model[1]
            self.features = self._model[2]
            self.cols_to_split = PTR_input_transform(
                self._model[3], self._model[4], self._model[5]
            )[0]
            self.labels = PTR_input_transform(
                self._model[3], self._model[4], self._model[5]
            )[1]
            self.tresholds = PTR_input_transform(
                self._model[3], self._model[4], self._model[5]
            )[2]

    def predict(self, df_portfolio: pd.DataFrame = None, **params) -> np.array:
        params: np.array = np.atleast_2d(np.asarray(self.params)).T
        predictions: np.array = df_portfolio.values @ params
        return predictions
