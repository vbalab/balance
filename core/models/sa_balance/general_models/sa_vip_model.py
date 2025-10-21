import pandas as pd
import numpy as np
from copy import deepcopy

# sys.path.append('../../../..')
from core.models.newbusiness.simple_adapters import (
    SimpleDataLoader,
    SimpleModelTrainer,
    SimpleModelAdapter,
)
from core.upfm.commons import (
    DataLoader,
    ModelTrainer,
    BaseModel,
    _REPORT_DT_COLUMN,
    ModelInfo,
    ForecastContext,
    ModelMetaInfo,
    ModelContainer,
)
from core.models.utils import (
    verify_data,
    gen_sa_product_balance_model_name,
    calculate_sa_model_features,
)
from datetime import datetime
from typing import Dict, Any, Tuple, List
from prophet import Prophet

# from orbit.models import DLT


CONFIG = {
    # fixed
    "features": [],
    "full_features": [],
    "target": [
        "SA_avg_balance_[general]_[vip]",
    ],
    "estimator": Prophet,
    "table_name": "prod_dadm_alm_sbx.almde_fl_dpst_saving_accounts_monthly_avg_feature",  # fixed
    "table_date_col": _REPORT_DT_COLUMN,
    "default_start_date": datetime(2018, 1, 1),
    "model_name": gen_sa_product_balance_model_name(segment="vip"),
    "model_params": {
        "seasonality_mode": "multiplicative",
        "changepoint_prior_scale": 1,
        "seasonality_prior_scale": 10,
    },
    "outliers": ["2022-02-28", "2022-03-31"],
    "prediction_type": "general_avg_balance",
}


# все выше пофиксил


class SaBalanceVipModel:
    def __init__(self, config=CONFIG):
        for key, value in config.items():
            setattr(self, key, value)

    #         self.estimator = self.estimator(response_col = self.target[0],
    #             date_col = _REPORT_DT_COLUMN,
    #             regressor_col = self.full_features, **self.model_params)

    def _generate_features(self, X):
        X_exog = pd.DataFrame(index=X.index)

        return X_exog.reset_index()

    def _generate_features_to_predict(self, X):
        if (
            X.index[0] - self.default_start_date
        ).days <= 31:  # Для случая прогноза in sample
            X = self._generate_features(X)
        elif (X.index[0] - self.last_fited_date).days > 40:
            raise ValueError(
                "More than 40 days between last train date (or last predict date) and current first predict date"
            )
        elif (X.index[0] - self.last_fited_date).days < 0:
            raise ValueError(
                "Less than 0 days between last train date (or last predict date) and current first predict date"
            )
        else:
            self.last_fited_date = X.index[-1]
            X = self._generate_features(X)

        return X

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        # учесть переименование колонок
        # учестть дамми
        # по аналогии с ТС
        verify_data(X, self.features, y, self.target)
        self.last_train_date = X.index[-1]
        self.X_stack = X.iloc[-1:, :]
        X = self._generate_features(X)
        y = y.reset_index()
        X = X.reset_index()
        train_df = X.merge(y, on="report_dt", how="left")
        # переименовывам под prophet
        train_df = train_df.rename(
            columns={self.table_date_col: "ds", self.target[0]: "y"}
        )
        # заменяем выбросы (вместо dummy)
        for outlier in self.outliers:
            train_df.loc[(train_df.ds == outlier), "y"] = None
        self.estimator = self.estimator(**self.model_params)
        for regressor in self.full_features:
            self.estimator.add_regressor(regressor, mode="multiplicative")
        train_df = train_df[train_df.ds.notna()]
        self.estimator.fit(train_df)

    def predict(self, X: pd.DataFrame):
        verify_data(X, self.features)
        index = X.index
        #         X = self._generate_features_to_predict(X)
        X = X.reset_index().rename(columns={X.index.name: "ds"})
        y_hat = self.estimator.predict(X)
        y_hat = pd.DataFrame(
            data=y_hat["yhat"].values, columns=self.target, index=index
        )
        y_hat.index.name = "report_dt"
        return y_hat


class SaBalanceVipDataLoader(SimpleDataLoader):
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
        target: pd.DataFrame = data[self.target]  # /1e09
        return {"features": features, "target": target}


class SaBalanceVipModelTrainer(SimpleModelTrainer):
    def __init__(self, config=CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = SaBalanceVipDataLoader()
        self.model = SaBalanceVipModel()


class SaBalanceVipModelAdapter(SimpleModelAdapter):
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
        df_date = forecast_context.model_data["features"].loc[forecast_dates, :]

        X = df_date.copy()
        pred = self._model_meta.predict(X)  # *1e09
        return pred


SaBalanceVip = ModelMetaInfo(
    model_name=CONFIG["model_name"],
    model_trainer=SaBalanceVipModelTrainer(),
    data_loader=SaBalanceVipDataLoader(),
    adapter=SaBalanceVipModelAdapter,
    segment="vip",
)
