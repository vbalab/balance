import pandas as pd
from datetime import datetime
from prophet import Prophet
from typing import Dict, Any
from pandas.tseries.offsets import MonthEnd

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
from core.models.utils import verify_data


CONFIG = {
    "features": ["VTB_ftp_rate_[90d]"],
    "full_features": ["VTB_ftp_rate_[90d]"],
    "target": [
        "balance_rub_[general]",
    ],
    "estimator": Prophet,
    "table_name": "prod_dadm_alm_sbx.almde_fl_dpst_current_accounts",
    "table_date_col": _REPORT_DT_COLUMN,
    "default_start_date": datetime(2011, 10, 31),
    "model_name": "current_accounts_model",
    "model_params": {
        "changepoints": ["2015-01-01", "2020-03-01", "2022-03-01"],
        "changepoint_prior_scale": 0.5,
        "changepoint_range": 1,
    },
    "add_dummies": {
        "covid_dummy": ["2020-03-31", "2022-02-28"],
        "svo_dummy": ["2022-02-28", (datetime.now() + MonthEnd()).strftime("%Y-%m-%d")],
    },
    "prediction_type": "general_avg_balance",
}


class CurrentAccountsModel:
    def __init__(self, config=CONFIG):
        for key, value in config.items():
            setattr(self, key, value)

    def _generate_features(self, X, fit=True):
        X_exog = pd.DataFrame(index=X.index)

        X_exog.loc[:, "VTB_ftp_rate_[90d]"] = X.loc[:, "VTB_ftp_rate_[90d]"]

        for dummy_name, dates in self.add_dummies.items():
            dates = pd.to_datetime(dates)
            if fit:
                if dates[0] <= X.index[-1]:
                    X_exog.loc[:, dummy_name] = (
                        (X.index >= dates[0]) & (X.index <= dates[1])
                    ).astype("int")
            else:
                if dates[0] < X.index[0]:
                    X_exog.loc[:, dummy_name] = (
                        (X.index >= dates[0]) & (X.index <= dates[1])
                    ).astype("int")

        return X_exog

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
            X = self._generate_features(X, fit=False)

        return X

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        verify_data(X, self.features, y, self.target)

        self.last_fited_date = X.index[-1]

        X = self._generate_features(X)
        train_df = pd.concat([X, y / 1e09], axis=1)
        train_df = train_df.reset_index().rename(
            columns={train_df.index.name: "ds", self.target[0]: "y"}
        )

        self.model_params["changepoints"] = [
            cp
            for cp in self.model_params["changepoints"]
            if X.index[0] <= pd.to_datetime(cp) <= X.index[-1]
        ]
        self.estimator = self.estimator(**self.model_params)

        for regressor in X.columns:
            self.estimator.add_regressor(regressor)

        self.estimator.fit(train_df)

    def predict(self, X: pd.DataFrame):
        verify_data(X, self.features)
        index = X.index
        X = (
            self._generate_features_to_predict(X)
            .reset_index()
            .rename(columns={X.index.name: "ds"})
        )
        y_hat = self.estimator.predict(X)
        y_hat = (
            pd.DataFrame(data=y_hat["yhat"].values, columns=self.target, index=index)
            * 1e09
        )
        y_hat.index.name = "report_dt"
        return y_hat


class CurrentAccountsDataLoader(SimpleDataLoader):
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


class CurrentAccountsModelTrainer(SimpleModelTrainer):
    def __init__(self, config=CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = CurrentAccountsDataLoader()
        self.model = CurrentAccountsModel()


class CurrentAccountsModelAdapter(SimpleModelAdapter):
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

        X = df_date[self.features]
        pred = self._model_meta.predict(X)  # *1e09
        return pred


CurrentAccountsBalance = ModelMetaInfo(
    model_name=CONFIG["model_name"],
    model_trainer=CurrentAccountsModelTrainer(),
    data_loader=CurrentAccountsDataLoader(),
    adapter=CurrentAccountsModelAdapter,
    segment=None,
)
