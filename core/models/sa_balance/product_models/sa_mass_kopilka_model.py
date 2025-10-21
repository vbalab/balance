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
from core.models.utils import gen_sa_product_balance_model_name
from datetime import datetime
from typing import Dict, Any

from sktime.forecasting.naive import NaiveForecaster


CONFIG = {
    "features": None,
    "full_features": None,
    "target": [
        "SA_avg_balance_[kopilka]_[mass]",
    ],
    "estimator": NaiveForecaster(strategy="mean", window_length=3),
    "table_name": "dadm_alm_sbx.saving_accounts_monthly_feature_table",
    "table_date_col": _REPORT_DT_COLUMN,
    "default_start_date": datetime(2019, 3, 1),
    "model_name": gen_sa_product_balance_model_name(product="kopilka", segment="mass"),
    "prediction_type": "kopilka_avg_balance",
}


class SaKopilkaMassModel:
    def __init__(self, config=CONFIG):
        for key, value in config.items():
            setattr(self, key, value)

    def _generate_features(self, X):
        pass

    def _generate_features_to_predict(self, X):
        pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        # verify_data(X, self.features, y, self.target)
        self.estimator.fit(y.asfreq("M"))

    def predict(self, X: pd.DataFrame):
        # verify_data(X, self.features)
        y_hat = self.estimator.predict(X)
        y_hat = pd.DataFrame(data=y_hat.values, columns=self.target, index=X)
        y_hat.index.name = "report_dt"
        return y_hat


class SaKopilkaMassDataLoader(SimpleDataLoader):
    def __init__(self, config=CONFIG):
        super().__init__(config)

    def get_prediction_data(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        pass


class SaKopilkaMassModelTrainer(SimpleModelTrainer):
    def __init__(self, config=CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = SaKopilkaMassDataLoader()
        self.model = SaKopilkaMassModel()


class SaKopilkaMassModelAdapter(SimpleModelAdapter):
    def __init__(
        self, model_info_: ModelInfo, filepath_or_buffer, config: dict = CONFIG
    ):
        super().__init__(model_info_, filepath_or_buffer, config)

    def predict(
        self,
        forecast_context: ForecastContext,
        portfolio: pd.DataFrame = None,
        **params
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


SaKopilkaMass = ModelMetaInfo(
    model_name=CONFIG["model_name"],
    model_trainer=SaKopilkaMassModelTrainer(),
    data_loader=SaKopilkaMassDataLoader(),
    adapter=SaKopilkaMassModelAdapter,
    segment="mass",
)
