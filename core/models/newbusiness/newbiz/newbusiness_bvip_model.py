import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from core.models.utils import (
    generate_svo_flg,
    verify_data,
    gen_newbiz_model_name,
    calculate_max_available_rate,
)
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
from datetime import datetime
from typing import Dict, Any

CONFIG = {
    "features": [
        "rate_sa_weighted",
        "VTB_max_rate_available_[vip]_[r1s1]",  #
        "VTB_max_rate_available_[vip]_[r0s0]",  #
        "VTB_weighted_rate_[vip]_[r0s0]",
        "VTB_max_rate_available_[vip]",
        "SBER_max_rate",
        "VTB_weighted_rate_[vip]",
        "plan_close_[vip]",
    ],
    "full_features": [
        "spread_[rate_sa_weighted-VTB_max_rate_available_[vip]_[r1s1]]",
        "spread_[rate_sa_weighted-VTB_weighted_rate_[vip]_[r0s0]]",
        "VTB_max_rate_available_[vip]_[r0s0]_delta1",
        "VTB_max_rate_available_[vip]_pct_change1",
        "svo_flg_feb",
        "svo_flg",
        "anomal_december",
        "prod_[plan_close*VTB_max_rate]",
        "anomal_september",
    ],
    "target": [
        "y_inflow_[bvip]",
    ],
    "estimator": LinearRegression(),
    "min_model_predict": 0.0,
    "table_name": "prod_dadm_alm_sbx.almde_fl_dpst_newbusiness_model_features",
    "table_date_col": _REPORT_DT_COLUMN,
    "default_start_date": datetime(2014, 2, 1),
    "model_name": gen_newbiz_model_name(segment="bvip"),
}


class NewbusinessBvipModel:
    def __init__(self, config=CONFIG):
        for key, value in config.items():
            setattr(self, key, value)

    SP_NAME = "spread_[VTB_weighted_rate_SBER_max_rate]"

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _generate_features(self, X):
        X_exog = pd.DataFrame(index=X.index)

        X_exog.loc[
            :, "spread_[rate_sa_weighted-VTB_max_rate_available_[vip]_[r1s1]]"
        ] = (
            X.loc[:, "rate_sa_weighted"]
            - X.loc[:, "VTB_max_rate_available_[vip]_[r1s1]"]
        )
        X_exog.loc[
            X.index < "2020-05-01",
            "spread_[rate_sa_weighted-VTB_max_rate_available_[vip]_[r1s1]]",
        ] = 0.0

        X_exog.loc[:, "spread_[rate_sa_weighted-VTB_weighted_rate_[vip]_[r0s0]]"] = (
            X.loc[:, "rate_sa_weighted"] - X.loc[:, "VTB_weighted_rate_[vip]_[r0s0]"]
        )
        X_exog.loc[
            X.index < "2020-05-01",
            "spread_[rate_sa_weighted-VTB_weighted_rate_[vip]_[r0s0]]",
        ] = 0.0

        # Добавляем функцию активации при спреде меньше нуля
        X_exog.loc[
            X_exog["spread_[rate_sa_weighted-VTB_weighted_rate_[vip]_[r0s0]]"] < 0,
            "spread_[rate_sa_weighted-VTB_weighted_rate_[vip]_[r0s0]]",
        ] = X_exog.loc[
            X_exog["spread_[rate_sa_weighted-VTB_weighted_rate_[vip]_[r0s0]]"] < 0,
            "spread_[rate_sa_weighted-VTB_weighted_rate_[vip]_[r0s0]]",
        ].apply(
            lambda x: self._sigmoid(x)
        )

        X_exog.loc[:, "prod_[plan_close*VTB_max_rate]"] = (
            abs(X.loc[:, "plan_close_[vip]"]) * X.loc[:, "VTB_max_rate_available_[vip]"]
        )

        X_exog.loc[:, "VTB_max_rate_available_[vip]_[r0s0]_delta1"] = (
            X.loc[:, "VTB_max_rate_available_[vip]_[r0s0]"].diff(1).fillna(0)
        )
        X_exog.loc[:, "VTB_max_rate_available_[vip]_pct_change1"] = (
            X.loc[:, "VTB_max_rate_available_[vip]"].pct_change(1).fillna(0)
        )

        X_exog.loc[:, "svo_flg_feb"] = (X.index == "2022-02-28").astype(float)
        X_exog = generate_svo_flg(X_exog)
        X_exog.loc[:, "anomal_december"] = (X.index == "2021-12-31").astype(float)
        X_exog.loc[:, "anomal_september"] = (X.index == "2022-09-30").astype(float)

        X_exog.loc[:, self.SP_NAME] = (
            X.loc[:, "VTB_weighted_rate_[vip]"] - X.loc[:, "SBER_max_rate"]
        )

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
            self.X_stack = X_added.iloc[-1:, :]
            X_added = self._generate_features(X_added)
            X = X_added.iloc[1:, :]

        return X

    def fit(self, X: pd.DataFrame, y):
        verify_data(X, self.features)
        self.X_stack = X.iloc[-1:, :]
        X = self._generate_features(X)

        if isinstance(y, pd.DataFrame):
            y = y.fillna(0.0)
            self.estimator.fit(X[self.full_features], np.ravel(y.values))
        else:
            self.estimator.fit(X[self.full_features], y)

    # добавляем корректирующие блоки перед предиктом

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _f_act_vip(self, x):
        if x < 1:
            return max(
                0.1,
                2
                * self._sigmoid(
                    (x / 1.9) ** 2 * np.sign(x) + (x / 3) ** 4 * np.sign(x)
                ),
            )
        else:
            return (
                5
                * self._sigmoid((x / 1.5) ** 2 * np.sign(x) + (x / 3) ** 4 * np.sign(x))
                - 1.9175
            )

    def _correct_pred_feature(self, df, y):
        coef = df[self.SP_NAME].apply(lambda x: self._f_act_vip(x))

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


class NewbusinessBvipDataLoader(SimpleDataLoader):
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


class NewbusinessBvipModelTrainer(SimpleModelTrainer):
    def __init__(self, config=CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = NewbusinessBvipDataLoader()
        self.model = NewbusinessBvipModel()


class NewbusinessBvipModelAdapter(SimpleModelAdapter):
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
            forecast_dates, "VTB_max_rate_available_[vip]_[r1s1]"
        ] = calculate_max_available_rate(df_date, segment="vip", repl=1, sub=1)
        forecast_context.model_data["features"].loc[
            forecast_dates, "VTB_max_rate_available_[vip]_[r0s0]"
        ] = calculate_max_available_rate(df_date, segment="vip", repl=0, sub=0)
        forecast_context.model_data["features"].loc[
            forecast_dates, "VTB_max_rate_available_[vip]"
        ] = calculate_max_available_rate(df_date, segment="vip")

        X = forecast_context.model_data["features"].loc[forecast_dates, self.features]
        pred = self._model_meta.predict(X)
        return pred


NewbusinessBvip = ModelMetaInfo(
    model_name=CONFIG["model_name"],
    model_trainer=NewbusinessBvipModelTrainer(),
    data_loader=NewbusinessBvipDataLoader(),
    adapter=NewbusinessBvipModelAdapter,
    segment="bvip",
)
