from core.upfm.commons import DataLoader, _REPORT_DT_COLUMN
from datetime import datetime
import pandas as pd
from core.definitions import SCENARIO_COLUMNS_, PORTFOLIO_COLUMNS_
from typing import Dict, Any, Tuple
import pyspark.sql.functions as f
from pyspark.sql import Window
from core.models.utils import convert_decimals

CONFIG = {
    "scenario_cols": SCENARIO_COLUMNS_,
    "scenario_table_name": "prod_dadm_alm_sbx.almde_fl_dpst_scenario",
    "scenario_table_date_col": _REPORT_DT_COLUMN,
    "portfolio_table_name": "prod_dadm_alm_sbx.almde_fl_dpst_early_close",
    "portfolio_table_date_col": "report_dt",
    "portfolio_table_date_str_col": "report_month",
    "portfolio_need_columns": PORTFOLIO_COLUMNS_,
    "optional_nan_threshold": 3e9,
}


class ScenarioLoader(DataLoader):
    def __init__(self, config=CONFIG):
        for key, value in config.items():
            setattr(self, key, value)

    def get_maximum_train_range(self, spark) -> Tuple[datetime, datetime]:
        pass

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
        start_date = "-".join(str(start_date.date()).split("-")[:2])
        end_date = "-".join(str(end_date.date()).split("-")[:2])

        w = Window.partitionBy("gen_name")

        portfolio = (
            spark.sql(f"select * from {self.portfolio_table_name}")
            .where(
                (f.col(self.portfolio_table_date_str_col) >= start_date)
                & (f.col(self.portfolio_table_date_str_col) <= end_date)
            )
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
            .withColumn(
                "close_month",
                f.when(f.col("close_month") > "2050-01", "2050-01").otherwise(
                    f.col("close_month")
                ),
            )
            .withColumnRenamed(self.portfolio_table_date_col, _REPORT_DT_COLUMN)
            .select(self.portfolio_need_columns)
        )
        port = convert_decimals(portfolio).toPandas()

        port.loc[port.bucketed_period.isna(), "bucketed_period"] = (
            pd.to_datetime(port.close_month) - pd.to_datetime(port.open_month)
        ).dt.days // 30 + 1

        if (
            port.loc[port.optional_flg.isna()]["total_generation"].sum() / 1e9
            > self.optional_nan_threshold
        ):
            raise ValueError(
                f"Volume of NAN values in optionalilty higher than threshold value: {self.optional_nan_threshold}"
            )

        port.optional_flg.fillna(0, inplace=True)
        port[_REPORT_DT_COLUMN] = pd.to_datetime(port[_REPORT_DT_COLUMN])

        return port

    def get_portfolio(
        self, spark, report_date: datetime, params: Dict[str, Any] = None
    ) -> Dict[str, pd.DataFrame]:
        report_month = "-".join(str(report_date.date()).split("-")[:2])

        w = Window.partitionBy("gen_name")

        portfolio = (
            spark.sql(f"select * from {self.portfolio_table_name}")
            .where(f.col(self.portfolio_table_date_str_col) == report_month)
            .where(f.col("cur") == "RUR")
            # .where(f.col('drop_flg') == 0)
            .where(f.col("close_month") > f.col("report_month"))
            .where(f.col("total_generation") > 0)
            # .withColumn(
            #     self.binary_option_flg_name,
            #     f.when((f.col('optional_flg') == 0), 0).otherwise(1)
            # )
            .withColumn("max_total_generation", f.max("total_generation").over(w))
            .withColumn("max_SER_dinamic", f.max("SER_dinamic").over(w))
            .withColumn(
                "close_month",
                f.when(f.col("close_month") > "2050-01", "2050-01").otherwise(
                    f.col("close_month")
                ),
            )
            .withColumnRenamed(self.portfolio_table_date_col, _REPORT_DT_COLUMN)
            .select(self.portfolio_need_columns)
        )
        port = convert_decimals(portfolio).toPandas()

        port.loc[port.bucketed_period.isna(), "bucketed_period"] = (
            pd.to_datetime(port.close_month) - pd.to_datetime(port.open_month)
        ).dt.days // 30 + 1

        if (
            port.loc[port.optional_flg.isna()]["total_generation"].sum() / 1e9
            > self.optional_nan_threshold
        ):
            raise ValueError(
                f"Volume of NAN values in optionalilty higher than threshold value: {self.optional_nan_threshold}"
            )

        port.optional_flg.fillna(0, inplace=True)
        port[_REPORT_DT_COLUMN] = pd.to_datetime(port[_REPORT_DT_COLUMN])

        return port

    def get_scenario(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> pd.DataFrame:
        scenario_data = (
            spark.sql(f"select * from {self.scenario_table_name}")
            .select([self.scenario_table_date_col] + self.scenario_cols)
            .toPandas()
        )

        scenario_data = scenario_data.sort_values(
            by=self.scenario_table_date_col
        ).reset_index(drop=True)

        scenario_data = scenario_data[
            scenario_data[self.scenario_table_date_col].between(start_date, end_date)
        ]
        scenario_data = scenario_data.set_index(self.scenario_table_date_col)
        scenario_data.index.name = _REPORT_DT_COLUMN
        scenario_data = scenario_data[self.scenario_cols].astype(float)
        scenario_data.fillna(0.01, inplace=True)
        return scenario_data
