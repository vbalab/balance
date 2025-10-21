import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from core.models.utils import (
    generate_svo_flg,
    verify_data,
    gen_newbiz_model_name,
    calculate_max_weighted_rate,
)
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
from datetime import datetime
from typing import Dict, Any, Tuple, List

CONFIG = {
    "features": [
        "VTB_weighted_rate_[vip]",
        "plan_close_[vip]",
        "VTB_ftp_rate_[365d]",
        "VTB_max_weighted_rate_[vip]_[r0s0]",  #
        "VTB_max_weighted_rate_[vip]",  #
        "rate_sa_weighted",
        "VTB_ftp_rate_[180d]",
        "SBER_max_rate",
    ],
    "full_features": [
        "y_inflow_[svip]_shift_1",
        "VTB_max_weighted_rate_delta1",
        "spread_[VTB_weighted_rate-VTB_[365d]_ftp_rate]",
        "spread_[rate_sa_weighted-VTB_max_weighted_rate_[r0s0]]",
        "pct_change1_VTB_[180d]_ftp_rate",
        "plan_close_sum_delta",
        "svo_flg_feb",
        "svo_flg",
        "anomal_december",
        "prod_[plan_close*VTB_max_rate]",
        "anomal_september",
    ],
    "target": [
        "y_inflow_[svip]",
    ],
    "estimator": LinearRegression(),
    "min_model_predict": 10e9,
    "table_name": "prod_dadm_alm_sbx.almde_fl_dpst_newbusiness_model_features",
    "table_date_col": _REPORT_DT_COLUMN,
    "default_start_date": datetime(2014, 2, 1),
    "model_name": gen_newbiz_model_name(segment="svip"),
}


class NewbusinessSvipModel:
    def __init__(self, config=CONFIG):
        for key, value in config.items():
            setattr(self, key, value)

    SP_NAME = "spread_[VTB_weighted_rate_SBER_max_rate]"

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _generate_exog_features(self, X):
        X_exog = pd.DataFrame(index=X.index)

        X_exog.loc[:, "VTB_max_weighted_rate_delta1"] = (
            X.loc[:, "VTB_max_weighted_rate_[vip]"].diff(1).fillna(0)
        )

        X_exog.loc[:, "spread_[VTB_weighted_rate-VTB_[365d]_ftp_rate]"] = (
            X.loc[:, "VTB_weighted_rate_[vip]"] - X.loc[:, "VTB_ftp_rate_[365d]"]
        )

        X_exog.loc[:, "spread_[rate_sa_weighted-VTB_max_weighted_rate_[r0s0]]"] = (
            X.loc[:, "rate_sa_weighted"]
            - X.loc[:, "VTB_max_weighted_rate_[vip]_[r0s0]"]
        )
        X_exog.loc[
            X.index < "2020-05-01",
            "spread_[rate_sa_weighted-VTB_max_weighted_rate_[r0s0]]",
        ] = 0.0

        # Добавляем функцию активации при спреде больше нуля
        X_exog.loc[
            X_exog["spread_[rate_sa_weighted-VTB_max_weighted_rate_[r0s0]]"] < 0,
            "spread_[rate_sa_weighted-VTB_max_weighted_rate_[r0s0]]",
        ] = X_exog.loc[
            X_exog["spread_[rate_sa_weighted-VTB_max_weighted_rate_[r0s0]]"] < 0,
            "spread_[rate_sa_weighted-VTB_max_weighted_rate_[r0s0]]",
        ].apply(
            lambda x: self._sigmoid(x)
        )

        X_exog.loc[:, "prod_[plan_close*VTB_max_rate]"] = (
            abs(X.loc[:, "plan_close_[vip]"]) * X.loc[:, "VTB_max_weighted_rate_[vip]"]
        )

        X_exog.loc[:, "pct_change1_VTB_[180d]_ftp_rate"] = (
            X.loc[:, "VTB_ftp_rate_[180d]"].pct_change(1).fillna(0)
        )

        X_exog.loc[:, "plan_close_sum_delta"] = (
            X.loc[:, "plan_close_[vip]"].diff(1).fillna(0)
        )

        X_exog.loc[:, "svo_flg_feb"] = (X.index == "2022-02-28").astype(float)
        X_exog = generate_svo_flg(X_exog)
        X_exog.loc[:, "anomal_december"] = (X.index == "2021-12-31").astype(float)
        X_exog.loc[:, "anomal_september"] = (X.index == "2022-09-30").astype(float)

        X_exog.loc[:, self.SP_NAME] = (
            X.loc[:, "VTB_weighted_rate_[vip]"] - X.loc[:, "SBER_max_rate"]
        )

        X_exog.index.name = "report_dt"
        return X_exog

    def _generate_endog_features_train(self, y):
        X_endog = pd.DataFrame(index=y.index)
        X_endog.loc[:, "y_inflow_[svip]_shift_1"] = (
            y.loc[:, "y_inflow_[svip]"].shift(1).bfill()
        )
        X_endog.index.name = "report_dt"
        return X_endog

    def _generate_features_train(self, X, y):
        X_exog = self._generate_exog_features(X)
        X_endog = self._generate_endog_features_train(y)
        X = X_endog.merge(X_exog, on="report_dt", how="outer")
        X = X[self.full_features]
        return X

    def _generate_features_predict(self, X):
        X_added = pd.concat([self.X_stack, X])
        self.X_stack = X_added.iloc[-1:, :]
        X_added = self._generate_exog_features(X_added)
        X = X_added.iloc[1:, :]
        X.loc[:, "y_inflow_[svip]_shift_1"] = self.y_stack.iloc[-1].values
        X = X  # [self.full_features]
        X.index.name = "report_dt"
        return X

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        verify_data(X, self.features, y, self.target)
        self.X_stack = X.iloc[-1:, :]
        self.y_stack = y.iloc[-1:, :]
        X = self._generate_features_train(X, y)
        y = y[self.target].fillna(0.0)
        if isinstance(y, pd.DataFrame):
            self.estimator.fit(X, np.ravel(y.values))
        else:
            self.estimator.fit(X, y)

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
        y_hat_list = []
        for step in range(X.shape[0]):
            X_cur = X.iloc[step : (step + 1), :]
            # заполняем пропуски в случае задания крайнего сценария
            X_cur = self._generate_features_predict(X_cur).fillna(0)
            y_hat_raw = self.estimator.predict(X_cur[self.full_features])
            y_hat_raw = np.round(y_hat_raw, 2)
            y_hat_raw = np.maximum(y_hat_raw, self.min_model_predict)

            # корректируем
            y_hat_raw = self._correct_pred_feature(X_cur, y_hat_raw).values

            y_hat_step = pd.DataFrame(
                data=y_hat_raw, columns=self.target, index=[X.index[step]]
            )
            y_hat_step.index.name = "report_dt"
            self.y_stack = self.y_stack.append(y_hat_step)
            y_hat_list.append(y_hat_step)
        y_hat = pd.concat(y_hat_list)
        return y_hat


class NewbusinessSvipDataLoader(SimpleDataLoader):
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


class NewbusinessSvipModelTrainer(SimpleModelTrainer):
    def __init__(self, config=CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = NewbusinessSvipDataLoader()
        self.model = NewbusinessSvipModel()


class NewbusinessSvipModelAdapter(SimpleModelAdapter):
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
            forecast_dates, "VTB_max_weighted_rate_[vip]_[r0s0]"
        ] = calculate_max_weighted_rate(df_date, segment="vip", repl=0, sub=0)
        forecast_context.model_data["features"].loc[
            forecast_dates, "VTB_max_weighted_rate_[vip]"
        ] = calculate_max_weighted_rate(df_date, segment="vip")

        X = forecast_context.model_data["features"].loc[forecast_dates, self.features]
        pred = self._model_meta.predict(X)
        return pred


NewbusinessSvip = ModelMetaInfo(
    model_name=CONFIG["model_name"],
    model_trainer=NewbusinessSvipModelTrainer(),
    data_loader=NewbusinessSvipDataLoader(),
    adapter=NewbusinessSvipModelAdapter,
    segment="svip",
)
