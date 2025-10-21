import pandas as pd
import numpy as np
from datetime import datetime
from pandas.tseries.offsets import MonthEnd
from core.models.newbusiness.simple_adapters import SimpleDataLoader, SimpleModelTrainer
from pickle import dump, PickleError, load
import pyspark.sql.functions as f
from pyspark.sql import Window
from core.upfm.commons import (
    DataLoader,
    BaseModel,
    ModelInfo,
    ForecastContext,
    _REPORT_DT_COLUMN,
    ModelMetaInfo,
)
import sys
import os.path
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
from core.models.utils import convert_decimals, dt_convert, check_existence
from core.definitions import PORTFOLIO_COLUMNS_

CONFIG = {
    "model_name": "plan_close",
    "default_start_date": datetime(2014, 2, 1),
    "target": ["plan_close_[mass]", "plan_close_[priv]", "plan_close_[vip]"],
    "features": [_REPORT_DT_COLUMN, "close_month", "is_vip_or_prv"],
}


class PlanCloseModel:
    def __init__(self, config=CONFIG):
        for key, value in config.items():
            setattr(self, key, value)

    def fit(self, *args, **kwargs):
        pass

    def predict(
        self, portfolio: pd.DataFrame, start_date: datetime, end_date: datetime
    ):
        if ~np.isin(self.features, portfolio.columns).all():
            raise KeyError(f"DataFrame should contains columns {self.features}")
        df_grouped = (
            portfolio.groupby(self.features)[
                "total_generation", "renewal_balance_next_month"
            ]
            .sum()
            .reset_index()
        )

        if len(df_grouped[_REPORT_DT_COLUMN].unique()) != 1:
            raise ValueError(f"Porfolio should has only one {_REPORT_DT_COLUMN}")

        df_grouped.loc[df_grouped["close_month"] > "3000-01", "close_month"] = "2050-01"
        df_grouped.loc[:, "close_month"] = pd.to_datetime(
            df_grouped["close_month"]
        ) + MonthEnd(0)
        df_grouped = df_grouped[
            df_grouped["close_month"].between(start_date, end_date)
        ].reset_index(drop=True)

        df_grouped["plan_close"] = df_grouped["total_generation"].astype(
            float
        ) - df_grouped["renewal_balance_next_month"].astype(float)
        df_grouped.loc[:, "is_vip_or_prv"] = df_grouped.loc[:, "is_vip_or_prv"].astype(
            int
        )

        outflow = df_grouped.pivot_table(
            values="plan_close", index="close_month", columns=["is_vip_or_prv"]
        )
        outflow.columns = self.target
        outflow.index.name = _REPORT_DT_COLUMN
        return outflow


class PlanCloseModelTrainer(SimpleModelTrainer):
    def __init__(self, config=CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = PlanCloseDataLoader()
        self.model = PlanCloseModel()


class PlanCloseDataLoader(DataLoader):
    def __init__(self):
        self.portfolio_table_name = "prod_dadm_alm_sbx.almde_fl_dpst_early_close"
        self.portfolio_table_date_col = "report_dt"
        self.portfolio_table_date_str_col = "report_month"
        # self.binary_option_flg_name = 'opt_flg_binary'
        self.portfolio_need_columns = PORTFOLIO_COLUMNS_

    def get_maximum_train_range(self, spark) -> Tuple[datetime, datetime]:
        start_date = datetime(2010, 1, 31)
        end_date = None
        return (start_date, None)

    def get_training_data(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        return {"features": pd.DataFrame(), "target": pd.DataFrame()}

    def get_prediction_data(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        return {"features": pd.DataFrame()}

    def get_ground_truth(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        return {"target": pd.DataFrame()}

    def get_portfolio(
        self, spark, report_date: datetime, params: Dict[str, Any] = None
    ) -> Dict[str, pd.DataFrame]:
        report_month = "-".join(str(report_date.date()).split("-")[:2])
        print("report_month:", report_month)

        w = Window.partitionBy("gen_name")

        portfolio = (
            spark.sql(f"select * from {self.portfolio_table_name}")
            .where(f.col(self.portfolio_table_date_str_col) == report_month)
            .where(f.col("cur") == "RUB")
            # .where(f.col('drop_flg') == 0)
            .where(f.col("close_month") > f.col("report_month"))
            .where(f.col("total_generation") > 0)
            # .withColumn(
            #     self.binary_option_flg_name,
            #     f.when((f.col('optional_flg') == 0), 0).otherwise(1)
            # )
            .withColumn("max_total_generation", f.max("total_generation").over(w))
            .withColumn("max_SER_dinamic", f.max("SER_dinamic").over(w))
            # добапвляем поля по бакетам баланса и медианным знаечниям типов клиентов
            .withColumn("share_buckets_balance", f.max("share_buckets_balance").over(w))
            .withColumn("3_med_pr_null", f.max("3_med_pr_null").over(w))
            .withColumn("3_med_pr_PENSIONER", f.max("3_med_pr_PENSIONER").over(w))
            .withColumn("3_med_pr_SALARY", f.max("3_med_pr_SALARY").over(w))
            .withColumn("3_med_pr_STANDART", f.max("3_med_pr_STANDART").over(w))
            .withColumn(
                "close_month",
                f.when(f.col("close_month") > "2050-01", "2050-01").otherwise(
                    f.col("close_month")
                ),
            )
            #             .withColumnRenamed(self.portfolio_table_date_col, _REPORT_DT_COLUMN)
            .select(self.portfolio_need_columns)
        )
        port = convert_decimals(portfolio).toPandas()

        port.loc[port.bucketed_period.isna(), "bucketed_period"] = (
            pd.to_datetime(port.close_month) - pd.to_datetime(port.open_month)
        ).dt.days // 30 + 1

        return port


class PlanCloseModelAdapter(BaseModel):
    def predict(
        self,
        forecast_context: ForecastContext,
        portfolio: pd.DataFrame = None,
        **params,
    ) -> Any:
        forecast_dates = forecast_context.forecast_dates
        portfolio_dt = forecast_context.portfolio_dt
        portfolio: pd.DataFrame = forecast_context.model_data["portfolio"][portfolio_dt]
        start_date = forecast_dates[0]
        end_date = forecast_dates[-1]
        pred_df = self._model_meta.predict(portfolio, start_date, end_date)
        return pred_df


PlanClose = ModelMetaInfo(
    model_name=CONFIG["model_name"],
    model_trainer=PlanCloseModelTrainer(),
    data_loader=PlanCloseDataLoader(),
    adapter=PlanCloseModelAdapter,
)
