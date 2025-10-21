from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from copy import deepcopy
from core.models.utils import (
    verify_data,
    gen_opt_model_name,
    calculate_max_rate,
    calculate_max_available_rate,
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

CONFIG = {
    "features": [
        "VTB_ftp_rate_[90d]",
        "VTB_weighted_rate_[vip]_[r1s1]",
        "VTB_weighted_rate_[vip]_[r0s0]",
        "VTB_max_weighted_rate_[vip]_[r0s0]",
        "VTB_max_rate_available_[vip]_[r0s0]"  # ,
        #'VTB_weighted_rate_[vip]_[r0s0]_[90d]'
    ],
    "full_features": [
        "y_inflow_[r0s0]_shift_1",
        "y_inflow_[r1s1]_mean_last3",
        "y_inflow_[r1s0]_shift_3",
        #'spread_[90d]_rate_[r0s0]_[VTB_ftp-VTB_weighted]',
        "spread_[VTB_weighted_rate_[r0s0]-VTB_max_rate_available_[r0s0]]",
        "spread_max_rate_[r0s0]_[VTB_ftp-VTB_weighted]",
        "VTB_weighted_rate_[r1s1]_[rate_share]",
    ],
    "target": [
        "y_inflow_share_[svip]_[r0s0]",
        "y_inflow_share_[svip]_[r0s1]",
        "y_inflow_share_[svip]_[r1s0]",
        "y_inflow_share_[svip]_[r1s1]",
    ],
    "estimator": RandomForestRegressor(
        **{
            "bootstrap": False,
            "criterion": "squared_error",
            "max_features": 5,
            "min_samples_split": 5,
            "n_estimators": 50,
        }
    ),
    "table_name": "prod_dadm_alm_sbx.almde_fl_dpst_opt_structure_model_features",
    "table_date_col": _REPORT_DT_COLUMN,
    "default_start_date": datetime(2017, 1, 1),
    "model_name": gen_opt_model_name(segment="svip"),
}


class OptStructureSvipModel:
    """
    Класс модели структуры опциональности.
    На вход берет взвешенные ставки внутри типов опциональностей сегмента привилегия
    (Например, взвешенная ставка в сегменте привилегия по вкладам без опций пополнения и снятия)
    На выход выдает распределение притоков в 4 типа опциональностей по долям
    """

    def __init__(self, config=CONFIG):
        for key, value in config.items():
            setattr(self, key, value)

    def _generate_exog_features(self, X):
        X_exog = pd.DataFrame(index=X.index)
        #         X_exog.loc[:,'spread_[90d]_rate_[r0s0]_[VTB_ftp-VTB_weighted]'] = \
        #             X.loc[:,'VTB_ftp_rate_[90d]'] - X.loc[:,'VTB_weighted_rate_[vip]_[r0s0]_[90d]']

        X_exog.loc[
            :, "spread_[VTB_weighted_rate_[r0s0]-VTB_max_rate_available_[r0s0]]"
        ] = (
            X.loc[:, "VTB_weighted_rate_[vip]_[r0s0]"]
            - X.loc[:, "VTB_max_rate_available_[vip]_[r0s0]"]
        )

        X_exog.loc[:, "spread_max_rate_[r0s0]_[VTB_ftp-VTB_weighted]"] = (
            X.loc[:, "VTB_ftp_rate_[90d]"]
            - X.loc[:, "VTB_max_weighted_rate_[vip]_[r0s0]"]
        )

        X_exog.loc[:, "VTB_weighted_rate_[r1s1]_[rate_share]"] = X.loc[
            :, "VTB_weighted_rate_[vip]_[r1s1]"
        ] / (
            X.loc[:, "VTB_weighted_rate_[vip]_[r1s1]"]
            + X.loc[:, "VTB_weighted_rate_[vip]_[r0s0]"]
        )
        X_exog.index.name = "report_dt"
        return X_exog

    def _generate_endog_features_train(self, y):
        X_endog = pd.DataFrame(index=y.index)
        X_endog.loc[:, "y_inflow_[r0s0]_shift_1"] = (
            y.loc[:, "y_inflow_share_[svip]_[r0s0]"].shift(1).bfill()
        )
        X_endog.loc[:, "y_inflow_[r1s1]_mean_last3"] = (
            y.loc[:, "y_inflow_share_[svip]_[r1s1]"].shift(1).rolling(3).mean().bfill()
        )
        X_endog.loc[:, "y_inflow_[r1s0]_shift_3"] = (
            y.loc[:, "y_inflow_share_[svip]_[r1s0]"].shift(3).bfill()
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
        X = self._generate_exog_features(X)
        X.loc[:, "y_inflow_[r0s0]_shift_1"] = self.y_stack.loc[
            :, "y_inflow_share_[svip]_[r0s0]"
        ][-1]
        X.loc[:, "y_inflow_[r1s1]_mean_last3"] = self.y_stack.loc[
            :, "y_inflow_share_[svip]_[r1s1]"
        ][-3:].mean()
        X.loc[:, "y_inflow_[r1s0]_shift_3"] = self.y_stack.loc[
            :, "y_inflow_share_[svip]_[r1s0]"
        ][-3]
        X = X[self.full_features]
        X.index.name = "report_dt"
        return X

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        verify_data(X, self.features, y, self.target)
        X = self._generate_features_train(X, y)
        y = y[self.target].fillna(0.0)
        y.index.name = "report_dt"
        self.estimator.fit(X, y)
        self.y_stack = y.iloc[-3:, :]

    def predict(self, X: pd.DataFrame):
        verify_data(X, self.features)
        y_hat_list = []
        for step in range(X.shape[0]):
            X_cur = X.iloc[step : (step + 1), :]
            # для обработки граничных сценариев заполняем нулями
            X_cur = self._generate_features_predict(X_cur).fillna(0)
            y_hat_raw = self.estimator.predict(X_cur)
            y_hat_raw = y_hat_raw / y_hat_raw.sum()
            y_hat_step = pd.DataFrame(
                data=y_hat_raw, columns=self.target, index=[X.index[step]]
            )
            y_hat_step.index.name = "report_dt"
            self.y_stack = self.y_stack.append(y_hat_step)
            y_hat_list.append(y_hat_step)
        y_hat = pd.concat(y_hat_list)
        return y_hat


class OptStructureSvipDataLoader(SimpleDataLoader):
    def __init__(self, config=CONFIG):
        super().__init__(config)


class OptStructureSvipModelTrainer(SimpleModelTrainer):
    def __init__(self, config=CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = OptStructureSvipDataLoader()
        self.model = OptStructureSvipModel()


class OptStructureSvipModelAdapter(SimpleModelAdapter):
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
        max_rate_vip_r0s0 = calculate_max_rate(df_date, segment="vip", repl=0, sub=0)
        max_available_rate_vip_r0s0 = calculate_max_available_rate(
            df_date, segment="vip", repl=0, sub=0
        )
        forecast_context.model_data["features"].loc[
            forecast_dates, "VTB_max_weighted_rate_[vip]_[r0s0]"
        ] = max_rate_vip_r0s0
        forecast_context.model_data["features"].loc[
            forecast_dates, "VTB_max_rate_available_[vip]_[r0s0]"
        ] = max_available_rate_vip_r0s0

        X = forecast_context.model_data["features"].loc[forecast_dates, self.features]
        pred = self._model_meta.predict(X)
        return pred


OptStructureSvip = ModelMetaInfo(
    model_name=CONFIG["model_name"],
    model_trainer=OptStructureSvipModelTrainer(),
    data_loader=OptStructureSvipDataLoader(),
    adapter=OptStructureSvipModelAdapter,
    segment="svip",
)
