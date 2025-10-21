import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from core.models.utils import generate_svo_flg, verify_data, gen_newbiz_model_name
from core.models.newbusiness.simple_adapters import (
    SimpleDataLoader,
    SimpleModelTrainer,
    SimpleModelAdapter,
)
from core.upfm.commons import (
    DataLoader,
    BaseModel,
    _REPORT_DT_COLUMN,
    ModelInfo,
    ForecastContext,
    ModelMetaInfo,
    ModelContainer,
)
from core.models.utils import (
    calculate_weighted_ftp_rate,
    calculate_max_rate,
    calculate_max_weighted_rate,
)
from datetime import datetime
from typing import Dict, Any, Tuple, List


CONFIG = {
    "features": [
        "VTB_weighted_rate_[mass]_[r1s1]",
        "VTB_weighted_rate_[mass]_[r0s0]",
        "VTB_max_rate_[mass]",
        "rate_sa_weighted",
        "VTB_weighted_rate_[mass]",
        "VTB_max_weighted_rate_[mass]",
        "plan_close_[mass]",
        "VTB_weighted_ftp_rate_[mass]",
        "SBER_max_rate",
    ],
    "full_features": [
        "spread_[VTB_weighted_rate[r1s1]_VTB_weighted_rate[r0s0]]",
        "VTB_max_rate_pct_change6_[mass]",
        "spread_[rate_sa_weighted-VTB_weighted_rate]",
        "VTB_max_weighted_rate_pct_change1_[mass]",
        "prod_[plan_close*VTB_max_rate]",
        "svo_flg",
        "december_flg",
        "spread_[VTB_weighted_ftp_rate_VTB_weighted_rate[r0s0]]",
        "new_trend",
    ],
    "target": [
        "y_inflow_[mass]",
    ],
    "estimator": LinearRegression(),
    "min_model_predict": 10e9,
    "table_name": "prod_dadm_alm_sbx.almde_fl_dpst_newbusiness_model_features",
    "table_date_col": _REPORT_DT_COLUMN,
    "default_start_date": datetime(2014, 1, 1),
    "model_name": gen_newbiz_model_name(segment="mass"),
}


class NewbusinessMassModel:
    def __init__(self, config=CONFIG):
        for key, value in config.items():
            setattr(self, key, value)

    SP_NAME = "spread_[VTB_weighted_rate_SBER_max_rate]"

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _generate_features(self, X):
        X_exog = pd.DataFrame(index=X.index)

        # X_exog['rate_sa_weighted'] = X_exog['rate_sa_weighted'].fillna(method='bfill')

        X_exog.loc[:, "december_flg"] = (X.index.month == 12).astype(float)

        X_exog.loc[:, "VTB_max_rate_pct_change6_[mass]"] = (
            X.loc[:, "VTB_max_rate_[mass]"].pct_change(6).fillna(0)
        )

        X_exog.loc[:, "prod_[plan_close*VTB_max_rate]"] = (
            abs(X.loc[:, "plan_close_[mass]"]) * X.loc[:, "VTB_max_rate_[mass]"]
        )

        X_exog.loc[:, "VTB_max_weighted_rate_pct_change1_[mass]"] = (
            X.loc[:, "VTB_max_weighted_rate_[mass]"].pct_change(1).fillna(0)
        )

        X_exog.loc[:, "spread_[rate_sa_weighted-VTB_weighted_rate]"] = (
            X.loc[:, "rate_sa_weighted"] - X.loc[:, "VTB_weighted_rate_[mass]"]
        )

        X_exog.loc[:, "spread_[rate_sa_weighted-VTB_weighted_rate]"] = X_exog.loc[
            :, "spread_[rate_sa_weighted-VTB_weighted_rate]"
        ].fillna(method="bfill")
        X_exog.loc[
            X_exog.index < "2019-05-01", "spread_[rate_sa_weighted-VTB_weighted_rate]"
        ] = 0.0

        # Добавляем функцию активации при спреде меньше нуля
        X_exog.loc[
            X_exog["spread_[rate_sa_weighted-VTB_weighted_rate]"] < 0,
            "spread_[rate_sa_weighted-VTB_weighted_rate]",
        ] = X_exog.loc[
            X_exog["spread_[rate_sa_weighted-VTB_weighted_rate]"] < 0,
            "spread_[rate_sa_weighted-VTB_weighted_rate]",
        ].apply(
            lambda x: self._sigmoid(x)
        )

        X_exog.loc[:, "spread_[VTB_weighted_ftp_rate_VTB_weighted_rate[r0s0]]"] = (
            X.loc[:, "VTB_weighted_ftp_rate_[mass]"]
            - X.loc[:, "VTB_weighted_rate_[mass]_[r0s0]"]
        )

        X_exog.loc[:, "spread_[VTB_weighted_rate[r1s1]_VTB_weighted_rate[r0s0]]"] = (
            X.loc[:, "VTB_weighted_rate_[mass]_[r1s1]"]
            - X.loc[:, "VTB_weighted_rate_[mass]_[r0s0]"]
        ).fillna(0)

        X_exog.loc[:, self.SP_NAME] = (
            X.loc[:, "VTB_weighted_rate_[mass]"] - X.loc[:, "SBER_max_rate"]
        )

        X_exog = generate_svo_flg(X_exog)

        new_trend = X_exog.index >= "2022-12-31"
        X_exog["new_trend"] = new_trend.astype(float)

        anomal_september = X_exog.index > "2023-09-30"
        X_exog["anomal_september"] = anomal_september.astype(float)

        X_exog = X_exog[self.full_features + [self.SP_NAME]]
        return X_exog

    def _generate_features_to_predict(self, X):
        if (
            X.index[0] - self.default_start_date
        ).days <= 31:  # Для случая прогноза in sample
            X = self._generate_features(X)
        elif (X.index[0] - self.X_stack.index[-1]).days > 40:
            raise ValueError(
                "More than 40 days between last train date (or last predict date) and current first predict date"
            )
        elif (X.index[0] - self.X_stack.index[-1]).days < 0:
            raise ValueError(
                "Less than 0 days between last train date (or last predict date) and current first predict date"
            )
        else:
            X_added = pd.concat([self.X_stack, X])
            self.X_stack = X_added.iloc[-6:, :]
            X_added = self._generate_features(X_added)
            X = X_added.iloc[6:, :]

        return X

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        verify_data(X, self.features, y, self.target)
        self.X_stack = X.iloc[-6:, :]
        X = self._generate_features(X)[self.full_features]
        if isinstance(y, pd.DataFrame):
            self.estimator.fit(X, np.ravel(y.values))
        else:
            self.estimator.fit(X, y)

    # добавляем корректирующие блоки перед предиктом

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _f_act_mass(self, x):
        if x < 1:
            return max(
                0.3,
                2
                * self._sigmoid(
                    ((x / 3) ** 2 * np.sign(x) + (x / 7) ** 4 * np.sign(x))
                ),
            )
        else:
            return (
                3 * self._sigmoid((x / 3) ** 2 * np.sign(x) + (x / 7) ** 4 * np.sign(x))
                - 1.0557 / 2
            )

    def _correct_pred_feature(self, df, y):
        coef = df[self.SP_NAME].apply(lambda x: self._f_act_mass(x))

        return coef * y

    def predict(self, X: pd.DataFrame):
        verify_data(X, self.features)
        X = self._generate_features_to_predict(X)
        y_hat = np.round(self.estimator.predict(X[self.full_features]), 2)
        y_hat = np.maximum(y_hat, self.min_model_predict)

        # корректируем
        y_hat = self._correct_pred_feature(X, y_hat).values

        y_hat = pd.DataFrame(data=y_hat, columns=self.target, index=X.index)
        y_hat.index.name = "report_dt"
        return y_hat


class NewbusinessMassDataLoader(SimpleDataLoader):
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


class NewbusinessMassModelTrainer(SimpleModelTrainer):
    def __init__(self, config=CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = NewbusinessMassDataLoader()
        self.model = NewbusinessMassModel()


class NewbusinessMassModelAdapter(SimpleModelAdapter):
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

        forecast_context.model_data["features"].loc[
            forecast_dates, "VTB_max_rate_[mass]"
        ] = calculate_max_rate(df_date, segment="mass")
        forecast_context.model_data["features"].loc[
            forecast_dates, "VTB_weighted_ftp_rate_[mass]"
        ] = calculate_weighted_ftp_rate(df_date, segment="mass")
        forecast_context.model_data["features"].loc[
            forecast_dates, "VTB_max_weighted_rate_[mass]"
        ] = calculate_max_weighted_rate(df_date, segment="mass")

        X = forecast_context.model_data["features"].loc[forecast_dates, self.features]
        pred = self._model_meta.predict(X)
        return pred


NewbusinessMass = ModelMetaInfo(
    model_name=CONFIG["model_name"],
    model_trainer=NewbusinessMassModelTrainer(),
    data_loader=NewbusinessMassDataLoader(),
    adapter=NewbusinessMassModelAdapter,
    segment="mass",
)
