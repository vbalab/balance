import numpy as np
import os
import pandas as pd
from sklearn.linear_model import Ridge
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
from sklearn.base import BaseEstimator


def gen_maturity_model_name(segment, repl, sub):
    if segment is None:
        segment = "segment"
    elif segment not in ["mass", "priv", "vip"]:
        raise ValueError("segment should be like ['mass', 'priv', 'vip'] or None")
    if repl is None and sub is None:
        repl, sub = -1, -1
    elif repl not in [0, 1] or sub not in [0, 1]:
        raise ValueError("repl and sub should be like [0,1] or None")
    return f"maturity_structure_{segment}_r{repl}s{sub}"


BASE_CONFIG = {
    "model_name": gen_maturity_model_name(None, None, None),
    "m": 1.5,
    "target": [
        "y_inflow_share[segment]_[opt]_[90d]",
        "y_inflow_share_[segment]_[opt]_[180d]",
        "y_inflow_share_[segment]_[opt]_[365d]",
        "y_inflow_share_[segment]_[opt]_[548d]",
        "y_inflow_share_[segment]_[opt]_[730d]",
        "y_inflow_share_[segment]_[opt]_[1095d]",
    ],
    "features": [
        "VTB_weighted_rate_[segment]_[opt]_[90d]",
        "VTB_weighted_rate_[segment]_[opt]_[180d]",
        "VTB_weighted_rate_[segment]_[opt]_[365d]",
        "VTB_weighted_rate_[segment]_[opt]_[548d]",
        "VTB_weighted_rate_[segment]_[opt]_[730d]",
        "VTB_weighted_rate_[segment]_[opt]_[1095d]",
    ],
    "weight_months": 6,
    "table_name": "prod_dadm_alm_sbx.almde_fl_dpst_maturity_structure_model_features",
    "table_date_col": _REPORT_DT_COLUMN,
    "default_start_date": datetime(2014, 1, 1),
    "inflow_threshold": 1e9,  # На суммарном притоке меньшего объема модель не обучается из-за нестабильности долей
}


# TODO: подбирать параметр m по кросс-валидации
class MaturityStructureBaseModel:
    M_MIN = 1.000001
    M_MAX = 2

    def __init__(self, config=BASE_CONFIG):
        for key, value in config.items():
            setattr(self, key, value)
        self.shares_list = []

    @classmethod
    def verify_m(cls, m):
        if (m < cls.M_MIN) or (m > cls.M_MAX):
            raise ValueError(
                f"Hyperparameter m must be between {cls.M_MIN} and {cls.M_MAX}"
            )

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, m):
        self.verify_m(m)
        self._m = m

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        X_model = X[self.features].fillna(0.01)
        y_model = y[self.target].fillna(0.0001)
        self.dynamic_weights_in_sum = (
            y_model.shift(1).rolling(self.weight_months).mean().bfill().values
        )
        self.last_weights = y_model.iloc[-self.weight_months :].mean().values
        max_rates = (np.max(X_model, axis=1) * self.m).values
        sum_rates = max_rates - np.sum(
            X_model.values * self.dynamic_weights_in_sum, axis=1
        )
        for i, rate in enumerate(self.features):
            feature = sum_rates / (max_rates - X_model[rate])
            model = Ridge(alpha=0.0, fit_intercept=False)
            self.shares_list.append(
                model.fit(feature.values.reshape(-1, 1), y_model[self.target[i]])
            )

    def predict(self, X: pd.DataFrame):
        # support only future forecasts
        sum_need_cols = all(np.isin(self.features, X.columns))
        if not sum_need_cols:
            raise ValueError("Features in train and features in predict are different!")

        X_model = X[self.features].fillna(0.01)
        max_rates = (np.max(X_model, axis=1) * self.m).values
        sum_rates = max_rates - np.sum(X_model.values * self.last_weights, axis=1)

        forecast = pd.DataFrame(columns=self.target)
        for i, rate in enumerate(self.features):
            # заполняем нулем в случае крайнего сценария - деления ноль на ноль
            feature = (sum_rates / (max_rates - X_model[rate])).fillna(0)
            y_hat = np.maximum(
                0.0, self.shares_list[i].predict(feature.values.reshape(-1, 1))
            )
            forecast[self.target[i]] = np.where(X[rate].isna(), 0, y_hat)
        forecast_sum = np.sum(forecast, axis=1)
        forecast_sum = np.where(forecast_sum > 0, forecast_sum, 0.001)
        forecast = forecast / forecast_sum.reshape(-1, 1)
        forecast.index = X.index
        forecast.index.name = _REPORT_DT_COLUMN
        return forecast


class MaturityStructureBaseDataLoader(DataLoader):
    def __init__(self, config=BASE_CONFIG):
        for key, value in config.items():
            setattr(self, key, value)
        # AdHoc Штука, если названия изменятся, нужно переписать
        self.absolute_inflows = [col.replace("share_", "") for col in self.target]

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
        if self.inflow_threshold is not None:
            abs_inflows = self._get_data(
                spark, start_date, end_date, columns=self.absolute_inflows
            ).sum(axis=1)
            data = data[abs_inflows > self.inflow_threshold]
        features: pd.DataFrame = data[self.features]
        target: pd.DataFrame = data[self.target]

        return {"features": features, "target": target}

    def get_prediction_data(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        features: pd.DataFrame = self._get_data(
            spark, start_date, end_date, columns=self.features
        )
        return {"features": features}

    def get_prediction_data_standalone(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        features: pd.DataFrame = self._get_data(
            spark, start_date, end_date, columns=self.features
        )
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
        return {"target": target.astype("float64")}

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


class MaturityStructureBaseModelTrainer(ModelTrainer):
    def __init__(self, config=BASE_CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = MaturityStructureBaseDataLoader()
        self.model = MaturityStructureBaseModel()

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


class MaturityStructureBaseModelAdapter(BaseModel):
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
        X = forecast_context.model_data["features"].loc[forecast_dates, self.features]
        pred = self._model_meta.predict(X)
        return pred
