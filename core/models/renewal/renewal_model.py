import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Tuple
from pickle import dump, PickleError
from datetime import datetime
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
from core.models.utils import sigmoid, dt_convert, check_existence, convert_decimals

from core.upfm.commons import (
    DataLoader,
    ModelTrainer,
    BaseModel,
    _REPORT_DT_COLUMN,
    ModelInfo,
    ForecastContext,
    ModelMetaInfo,
)

CONFIG = {
    "cl_target": ["renewal_label"],
    "regr_target": ["renewal_share"],
    "cl_features": [
        "bucketed_balance",
        "bucketed_period",
        "optional_flg",
        "is_vip_or_prv",
        "total_generation",
        "weight_rate",
        "weight_renewal_available_flg",
        "weight_renewal_cnt",
        "weight_close_plan_day",
    ],
    "regr_features": [
        "bucketed_balance",
        "bucketed_period",
        "optional_flg",
        "is_vip_or_prv",
        "total_generation",
        "weight_rate",
        #'weight_renewal_rate',
        "weight_renewal_available_flg",
        "weight_renewal_cnt",
        "weight_close_plan_day",
    ],
    "target": ["renewal_share"],
    "cl_estimator": CatBoostClassifier(iterations=200, verbose=False),
    "regr_estimator": CatBoostRegressor(iterations=1000, verbose=False),
    "table_name": "prod_dadm_alm_sbx.almde_fl_dpst_renewal_model",
    "table_date_col": _REPORT_DT_COLUMN,
    "default_start_date": datetime(2014, 1, 1),
    "model_name": "renewal_model",
    "portfolio_key": "portfolio",
}
CONFIG["features"] = list(set(CONFIG["cl_features"] + CONFIG["regr_features"]))


class RenewalModel:
    def __init__(self, config=CONFIG):
        for key, value in config.items():
            setattr(self, key, value)

    @staticmethod
    def _regr_weight_func(x: float) -> float:
        return sigmoid(np.log(x / 1e7)) * np.log(x / 10)

    def _prep_data_for_classification(self, df: pd.DataFrame) -> dict:
        df = df[df["total_generation"] > 1e4]
        df.loc[:, self.cl_target] = None
        df.loc[df.renewal_share == 0, self.cl_target] = 0
        df.loc[(df.renewal_share > 0), self.cl_target] = 1
        df_classif = (
            df[self.cl_features + self.cl_target].dropna().reset_index(drop=True)
        )
        fit_params = dict(
            X=df_classif.loc[:, self.cl_features],
            y=df_classif.loc[:, self.cl_target].values,
            sample_weight=df_classif.loc[:, "total_generation"],
        )
        return fit_params

    def _prep_data_for_regression(self, df: pd.DataFrame) -> dict:
        df = df[
            (df.renewal_share > 0)
            & (df.renewal_share < 1)
            & (df.total_generation > 1e4)
        ].reset_index(drop=True)
        df_regr = (
            df[self.regr_features + self.regr_target].dropna().reset_index(drop=True)
        )
        fit_params = dict(
            X=df_regr.loc[:, self.regr_features],
            y=df_regr.loc[:, self.regr_target].values,
            sample_weight=df_regr.loc[:, "total_generation"].apply(
                self._regr_weight_func
            ),
        )
        return fit_params

    def fit(self, df: pd.DataFrame) -> None:
        self.cl_estimator.fit(**self._prep_data_for_classification(df))
        self.regr_estimator.fit(**self._prep_data_for_regression(df))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        y_cl_hat = self.cl_estimator.predict(df[self.cl_features])
        y_cl_hat = np.where(
            (df.total_generation > 1e9) & (df.weight_renewal_available_flg > 0.1),
            1,
            y_cl_hat,
        )
        y_regr_hat = np.maximum(self.regr_estimator.predict(df[self.regr_features]), 0)
        y_regr_hat = np.where(
            df.total_generation > 1e8,
            np.minimum(y_regr_hat, df.weight_renewal_available_flg),
            y_regr_hat,
        )
        y_hat = y_cl_hat * y_regr_hat

        return y_hat


class RenewalDataLoader(DataLoader):
    def __init__(self, config=CONFIG):
        for key, value in config.items():
            setattr(self, key, value)

    def get_maximum_train_range(self, spark) -> Tuple[datetime, datetime]:
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

        return {"data": data}

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
        return {"data": data}

    def get_prediction_data_standalone(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        return self.get_prediction_data(spark, start_date, end_date)

    def get_ground_truth(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        pass

    def _get_data(
        self, spark, start_date: datetime, end_date: datetime, columns: list
    ) -> pd.DataFrame:
        features_df = spark.sql(f"select * from {self.table_name}").select(
            columns + [self.table_date_col]
        )
        features_df = convert_decimals(features_df).toPandas()
        features_df.loc[:, self.table_date_col] = pd.to_datetime(
            features_df.loc[:, self.table_date_col]
        )
        features_df = features_df.sort_values(by=self.table_date_col).reset_index(
            drop=True
        )

        # тут не оптимально - надо будет переделать, чтобы даты отсекались до toPandas
        # но надо быть аккуратным с краевыми случаями
        features_df = features_df[
            features_df[self.table_date_col].between(start_date, end_date)
        ]
        return features_df


class RenewalModelTrainer(ModelTrainer):
    def __init__(self, config=CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = RenewalDataLoader(config)
        self.model = RenewalModel()

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
        self.model.fit(self.training_data["data"])
        return self.model


class RenewalModelAdapter(BaseModel):
    def __init__(
        self, model_info_: ModelInfo, filepath_or_buffer, config=CONFIG
    ) -> None:
        super().__init__(model_info_, filepath_or_buffer)
        for key, value in config.items():
            setattr(self, key, value)

    def _split_portfolio(self, portfolio):
        close_mask = portfolio.close_month.isin(
            [dt.strftime("%Y-%m") for dt in self.forecast_dates]
        )
        return portfolio[close_mask], portfolio[~close_mask]

    def _make_prediction(self, portfolio):
        normal_port = portfolio[portfolio[self.features].isna().sum(axis=1) == 0]
        bad_port = portfolio[portfolio[self.features].isna().sum(axis=1) > 0]

        # Не совсем корректно так предсказывать, тк модель обучалась немного на других поколениях
        # Но, кажется, что разница не слишком критична (в будущем лучше исправить)
        renewal_share = self.model.predict(normal_port)
        normal_port.loc[:, "renewal_balance_next_month"] = (
            renewal_share * normal_port["total_generation"]
        )
        if bad_port.shape[0] > 0:
            bad_port.loc[:, "renewal_balance_next_month"] = 0.0
        portfolio = pd.concat([normal_port, bad_port])
        return portfolio

    def predict(
        self,
        forecast_context: ForecastContext,
        portfolio: pd.DataFrame = None,
        **params,
    ):
        self.forecast_dates = forecast_context.forecast_dates
        self.model_data = forecast_context.model_data

        portfolio = forecast_context.model_data["portfolio"][
            forecast_context.portfolio_dt
        ]
        close_portfolio, no_close_portfolio = self._split_portfolio(portfolio)
        close_portfolio = self._make_prediction(close_portfolio)
        no_close_portfolio.loc[:, "renewal_balance_next_month"] = 0.0
        portfolio = pd.concat([close_portfolio, no_close_portfolio])

        return portfolio.sort_index()


Renewal = ModelMetaInfo(
    model_name=CONFIG["model_name"],
    model_trainer=RenewalModelTrainer(),
    data_loader=RenewalDataLoader(),
    adapter=RenewalModelAdapter,
)
