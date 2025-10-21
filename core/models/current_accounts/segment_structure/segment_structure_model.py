import pandas as pd

# sys.path.append('../../../..')
from core.models.newbusiness.simple_adapters import (
    SimpleDataLoader,
    SimpleModelTrainer,
    SimpleModelAdapter,
)
from core.upfm.commons import (
    _REPORT_DT_COLUMN,
    ModelInfo,
    ForecastContext,
    ModelMetaInfo,
)
from core.definitions import DEFAULT_SEGMENTS_
from datetime import datetime
from typing import Dict, Any

from sktime.forecasting.naive import NaiveForecaster


TARGET_PREFIX = "CA_segment_share"

CONFIG = {
    "features": None,
    "full_features": None,
    "target": [
        "_".join([TARGET_PREFIX, "[mass]"]),
        "_".join([TARGET_PREFIX, "[priv]"]),
        "_".join([TARGET_PREFIX, "[vip]"]),
    ],
    "estimator": {
        "priv": NaiveForecaster(strategy="mean", window_length=3),
        "vip": NaiveForecaster(strategy="mean", window_length=3),
    },
    "table_name": "prod_dadm_alm_sbx.almde_fl_dpst_current_accounts",
    "table_date_col": _REPORT_DT_COLUMN,
    "default_start_date": datetime(2019, 3, 1),
    "model_name": "current_accounts_segment_structure_model",
    "prediction_type": "segment_structure",
}


class CaSegmentStructureModel:
    def __init__(self, config=CONFIG):
        for key, value in config.items():
            setattr(self, key, value)

    def _generate_features(self, X):
        pass

    def _generate_features_to_predict(self, X):
        pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        # verify_data(X, self.features, y, self.target)
        for segment in self.estimator.keys():
            self.estimator[segment] = self.estimator[segment].fit(
                y.loc[:, f"{TARGET_PREFIX}_[{segment}]"].asfreq("M")
            )

    def predict(self, X: pd.DataFrame):
        # verify_data(X, self.features)
        ca_shares_hat_list = []
        for segment in self.estimator.keys():
            y_hat = self.estimator[segment].predict(X)
            y_hat = pd.DataFrame(
                data=y_hat.values, columns=[f"{TARGET_PREFIX}_[{segment}]"], index=X
            )
            y_hat.index.name = "report_dt"
            ca_shares_hat_list.append(y_hat)

        rem_segment = DEFAULT_SEGMENTS_ - self.estimator.keys()
        if len(rem_segment) == 1:
            y_hat = 1 - pd.concat(ca_shares_hat_list, axis=1).sum(axis=1).values
            y_hat = pd.DataFrame(
                data=y_hat,
                columns=["{}_[{}]".format(TARGET_PREFIX, *rem_segment)],
                index=X,
            )
            y_hat.index.name = "report_dt"
            ca_shares_hat_list.append(y_hat)

        return pd.concat(ca_shares_hat_list, axis=1)


class CaSegmentStructureDataLoader(SimpleDataLoader):
    def __init__(self, config=CONFIG):
        super().__init__(config)

    def get_training_data(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        col_prefix = "balance_rub"
        target_cols = dict(
            zip(
                DEFAULT_SEGMENTS_,
                [f"{col_prefix}_[{segment}]" for segment in DEFAULT_SEGMENTS_],
            )
        )

        if isinstance(self.features, list):
            cols = list(target_cols.values()) + self.features
        data: pd.DataFrame = self._get_data(spark, start_date, end_date, columns=cols)

        if len(self.features) > 0:
            features: pd.DataFrame = data[self.features]
        else:
            features = pd.DataFrame(index=data.index)

        target: pd.DataFrame = pd.DataFrame(index=data.index)
        for segment, col in target_cols.items():
            target.loc[:, f"{TARGET_PREFIX}_[{segment}]"] = data.loc[:, col] / data.loc[
                :, list(target_cols.values())
            ].sum(axis=1)

        return {"features": features, "target": target}

    def get_prediction_data(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        pass

    def get_ground_truth(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        return {"target": pd.DataFrame()}  # Заглушка. TODO: Сделать нормальный метод.


class CaSegmentStructureModelTrainer(SimpleModelTrainer):
    def __init__(self, config=CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = CaSegmentStructureDataLoader()
        self.model = CaSegmentStructureModel()


class CaSegmentStructureModelAdapter(SimpleModelAdapter):
    def __init__(
        self, model_info_: ModelInfo, filepath_or_buffer, config: dict = CONFIG
    ):
        super().__init__(model_info_, filepath_or_buffer, config)

    def predict(
        self,
        forecast_context: ForecastContext,
        portfolio: pd.DataFrame = None,
        **params,
    ):
        forecast_dates = forecast_context.forecast_dates
        if self.features is not None:
            X = forecast_context.model_data["features"].loc[
                forecast_dates, self.features
            ]
        else:
            X = pd.DatetimeIndex(data=forecast_dates, freq="M", closed="right")
        pred = self._model_meta.predict(X)
        return pred


CaSegmentStructure = ModelMetaInfo(
    model_name=CONFIG["model_name"],
    model_trainer=CaSegmentStructureModelTrainer(),
    data_loader=CaSegmentStructureDataLoader(),
    adapter=CaSegmentStructureModelAdapter,
)
