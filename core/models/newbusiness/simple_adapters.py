import numpy as np
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple
from pickle import dump, PickleError, load
import pyspark.sql.functions as f
from core.upfm.commons import (
    DataLoader,
    ModelTrainer,
    BaseModel,
    _REPORT_DT_COLUMN,
    ModelInfo,
    ForecastContext,
)
from core.models.utils import dt_convert, check_existence
from typing import List


CONFIG_EXAMPLE = {
    "model_name": str,
    "target": List[str],
    "features": List[str],
    "table_name": str,
    "table_date_col": _REPORT_DT_COLUMN,
    "default_start_date": datetime(2014, 1, 1),
}


class SimpleDataLoader(DataLoader):
    def __init__(self, config):
        for key, value in config.items():
            setattr(self, key, value)
        if self.features is None:
            self.features = []

    def get_maximum_train_range(self, spark) -> Tuple[datetime, datetime]:
        # хранить ли названия колонок?
        features = spark.sql(f"""select * from {self.table_name}""")

        features_dropna = features.select(self.features + [self.table_date_col]).dropna(
            how="all"
        )

        feature_dates = features_dropna.select(
            f.min(f"{self.table_date_col}").alias("min_date"),
            f.max(f"{self.table_date_col}").alias("max_date"),
        ).collect()

        feature_min_date = feature_dates[0]["min_date"]
        feature_max_date = feature_dates[0]["max_date"]

        if isinstance(feature_min_date, datetime):
            min_date = max(feature_min_date, self.default_start_date)
            max_date = feature_max_date
            return min_date, max_date
        else:
            raise TypeError(f"column {self.table_date_col} is not a datetime column")

    def get_training_data(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        cols = self.features + self.target
        data: pd.DataFrame = self._get_data(spark, start_date, end_date, columns=cols)
        if len(self.features) > 0:
            features: pd.DataFrame = data[self.features]
        else:
            features = pd.DataFrame(index=data.index)
        target: pd.DataFrame = data[self.target]

        return {"features": features, "target": target}

    def get_prediction_data(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        if len(self.features) > 0:
            features: pd.DataFrame = self._get_data(
                spark, start_date, end_date, columns=self.features
            )
        else:
            dates = pd.date_range(
                start=start_date, end=end_date, freq="M", closed="right"
            )
            features = pd.DataFrame(index=dates)
            features.index.name = _REPORT_DT_COLUMN
        return {"features": features}

    def get_prediction_data_standalone(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        if len(self.features) > 0:
            features: pd.DataFrame = self._get_data(
                spark, start_date, end_date, columns=self.features
            )
        else:
            dates = pd.date_range(
                start=start_date, end=end_date, freq="M", closed="right"
            )
            features = pd.DataFrame(index=dates)
            features.index.name = _REPORT_DT_COLUMN
        return {"features": features}

    def get_ground_truth(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        if not hasattr(self, "ground_truth"):
            self._load_data(spark, columns=self.target, attr_name="ground_truth")

        target: pd.DataFrame = self.ground_truth[
            self.ground_truth[self.table_date_col].between(start_date, end_date)
        ]
        target = target.set_index(self.table_date_col)
        target.index.name = _REPORT_DT_COLUMN
        return {"target": target}

    def _get_data(
        self, spark, start_date: datetime, end_date: datetime, columns: list
    ) -> pd.DataFrame:
        features_df = (
            spark.sql(f"select * from {self.table_name}")
            .select(columns + [self.table_date_col])
            .toPandas()
        )
        features_df = features_df.sort_values(by=self.table_date_col).reset_index(
            drop=True
        )

        features_df = features_df[
            features_df[self.table_date_col].between(start_date, end_date)
        ]
        features_df = features_df.set_index(self.table_date_col)
        features_df.index.name = _REPORT_DT_COLUMN
        df = features_df[columns].astype(float)
        return df

    def _load_data(
        self,
        spark,
        start_date: datetime = None,
        end_date: datetime = None,
        columns: list = None,
        attr_name: str = "table_data",
    ):
        if start_date is None:
            start_date = self.default_start_date

        data = spark.table(self.table_name).filter(
            f.col(self.table_date_col) >= start_date.strftime("%Y-%m-%d")
        )

        if end_date:
            data = data.filter(
                f.col(self.table_date_col) <= end_date.strftime("%Y-%m-%d")
            )

        if columns:
            data = data.select(columns + [self.table_date_col])

        data = (
            data.toPandas().sort_values(by=self.table_date_col).reset_index(drop=True)
        )
        data.loc[:, self.table_date_col] = pd.to_datetime(data[self.table_date_col])
        # data.index.name = _REPORT_DT_COLUMN
        setattr(self, attr_name, data)


class SimpleModelTrainer(ModelTrainer):
    def __init__(self, config) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = SimpleDataLoader(config)
        self.model = BaseModel()

    def get_trained_model(
        self,
        spark,
        end_date: datetime,
        start_date: datetime = None,
        hyperparams: Dict[str, Any] = None,
    ) -> Any:
        if start_date is None:
            start_date = self.default_start_date

        return self._training(spark, end_date, start_date, hyperparams)

    def save_trained_model(
        self,
        spark,
        saving_path: str,
        end_date: datetime,
        start_date: datetime = None,
        overwrite: bool = True,
        hyperparams: Dict[str, Any] = None,
    ) -> str:
        if start_date is None:
            start_date = self.default_start_date

        fname = (
            f"{self.model_name}_{dt_convert(start_date)}_{dt_convert(end_date)}.pickle"
        )

        if check_existence(path=saving_path, name=fname, overwrite=overwrite):
            pickle_name = fname
            return pickle_name

        try:
            file = open(os.path.join(saving_path, fname), "wb")
            model_fit = self.get_trained_model(spark, end_date, start_date, hyperparams)
            dump(model_fit, file)
            pickle_name = fname
        except (OSError, PickleError, RecursionError) as e:
            print(e)
            pickle_name = None

        return pickle_name

    def _training(
        self,
        spark,
        end_date: datetime,
        start_date: datetime,
        hyperparams: Dict[str, Any],
    ) -> Any:
        self.training_data: Dict[str, pd.DataFrame] = self.dataloader.get_training_data(
            spark, start_date, end_date
        )
        self.model.fit(self.training_data["features"], self.training_data["target"])
        return self.model


class SimpleModelAdapter(BaseModel):
    def __init__(
        self, model_info_: ModelInfo, filepath_or_buffer, config: dict
    ) -> None:
        super().__init__(model_info_, filepath_or_buffer)
        for key, value in config.items():
            setattr(self, key, value)

    def predict(
        self,
        forecast_context: ForecastContext,
        portfolio: pd.DataFrame = None,
        **params,
    ) -> Any:
        forecast_dates = forecast_context.forecast_dates
        if self.features is not None:
            X = forecast_context.model_data["features"].loc[
                forecast_dates, self.features
            ]
        else:
            X = pd.DataFrame(index=forecast_dates)
        pred = self._model_meta.predict(X)
        return pred
