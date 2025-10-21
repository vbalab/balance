from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from copy import deepcopy
from core.models.utils import verify_data, gen_opt_model_name
from core.models.newbusiness.simple_adapters import (
    SimpleDataLoader,
    SimpleModelTrainer,
    SimpleModelAdapter,
)
from core.upfm.commons import (
    _REPORT_DT_COLUMN,
    ModelInfo,
    ModelMetaInfo,
)
from datetime import datetime

CONFIG = {
    "features": [
        "VTB_weighted_rate_[priv]_[r0s0]",
        "VTB_weighted_rate_[priv]_[r0s1]",
        "VTB_weighted_rate_[priv]_[r1s0]",
        "VTB_weighted_rate_[priv]_[r1s1]",
    ],
    "target": [
        "y_inflow_share_[priv]_[r0s0]",
        "y_inflow_share_[priv]_[r0s1]",
        "y_inflow_share_[priv]_[r1s0]",
        "y_inflow_share_[priv]_[r1s1]",
    ],
    "estimator": RandomForestRegressor(n_estimators=100, max_depth=None),
    "table_name": "prod_dadm_alm_sbx.almde_fl_dpst_opt_structure_model_features",
    "table_date_col": _REPORT_DT_COLUMN,
    "default_start_date": datetime(2016, 10, 1),
    "model_name": gen_opt_model_name(segment="priv"),
}


class OptStructurePrivModel:
    """
    Класс модели структуры опциональности.
    На вход берет взвешенные ставки внутри типов опциональностей сегмента привилегия
    (Например, взвешенная ставка в сегменте привилегия по вкладам без опций пополнения и снятия)
    На выход выдает распределение притоков в 4 типа опциональностей по долям
    """

    def __init__(self, config=CONFIG):
        for key, value in config.items():
            setattr(self, key, value)

    def _change_values(self, X, y=None):
        if y is not None:
            X.loc[
                y["y_inflow_share_[priv]_[r1s0]"] < 1e-5,
                "VTB_weighted_rate_[priv]_[r1s0]",
            ] = 0.0
            X.loc[
                y["y_inflow_share_[priv]_[r0s1]"] < 1e-5,
                "VTB_weighted_rate_[priv]_[r0s1]",
            ] = 0.0
        return X

    def _generate_features(self, X):
        X.loc[:, "rate_opt_diff"] = (
            X["VTB_weighted_rate_[priv]_[r1s1]"].diff(1).fillna(0)
        )
        X.loc[:, "spread"] = (
            X["VTB_weighted_rate_[priv]_[r0s0]"] - X["VTB_weighted_rate_[priv]_[r1s1]"]
        )
        X.loc[:, "spread_diff"] = X["spread"].diff(1).fillna(0)
        X.loc[:, "december_flg"] = (X.index.month == 12).astype(float)
        return X

    def _generate_features_to_predict(self, X):
        if (X.index[0] - self.last_row_X.index[-1]).days > 40:
            raise ValueError(
                "More than 40 days between last train date (or last predict date) and current first predict date"
            )
        if (X.index[0] - self.last_row_X.index[-1]).days < 0:
            raise ValueError(
                "Less than 0 days between last train date (or last predict date) and current first predict date"
            )
        X_added = pd.concat([self.last_row_X, X])
        X_added = self._generate_features(X_added)
        X = X_added.iloc[1:, :]
        return X

    # подумать нужен ли этот метод - слишком много путанницы
    # без него модель будет поддерживать начало иитеративных прогнозов только из даты t+1
    # где t - последняя дата обучающей выборки
    def _find_last_row(self, X):
        pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        verify_data(X, self.features, y, self.target)
        y = deepcopy(y[self.target].fillna(0))
        X = deepcopy(X[self.features].fillna(0))
        X = self._change_values(X, y)
        self.last_row_X = X.iloc[-1:, :]
        X = self._generate_features(X)
        self.estimator.fit(X, y)

    def predict(self, X: pd.DataFrame):
        verify_data(X, self.features)
        X = deepcopy(X[self.features].fillna(0))
        X = self._generate_features_to_predict(X)
        y_hat = np.round(self.estimator.predict(X), 4)
        y_hat = np.where(X[self.features].values < 1e-1, 0, y_hat)
        y_hat = y_hat / y_hat.sum(axis=1).reshape(-1, 1)
        y_hat = pd.DataFrame(data=y_hat, columns=self.target, index=X.index)
        y_hat.index.name = "report_dt"
        self.last_row_X = X.iloc[-1:, :]
        return y_hat


class OptStructurePrivDataLoader(SimpleDataLoader):
    def __init__(self, config=CONFIG):
        super().__init__(config)


class OptStructurePrivModelTrainer(SimpleModelTrainer):
    def __init__(self, config=CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = OptStructurePrivDataLoader()
        self.model = OptStructurePrivModel()


class OptStructurePrivModelAdapter(SimpleModelAdapter):
    def __init__(
        self, model_info_: ModelInfo, filepath_or_buffer, config: dict = CONFIG
    ):
        super().__init__(model_info_, filepath_or_buffer, config)


OptStructurePriv = ModelMetaInfo(
    model_name=CONFIG["model_name"],
    model_trainer=OptStructurePrivModelTrainer(),
    data_loader=OptStructurePrivDataLoader(),
    adapter=OptStructurePrivModelAdapter,
    segment="priv",
)
