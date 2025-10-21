import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from core.models.utils import (
    generate_svo_flg,
    verify_data,
    gen_newbiz_model_name,
    gaussian_kernel,
    parse_buckets_from_port,
    calc_model_bucket_share,
    calc_new_shares,
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
from core.definitions import *
from core.models.utils import (
    calculate_weighted_ftp_rate,
    calculate_max_rate,
    calculate_max_weighted_rate,
)
from datetime import datetime
from typing import Dict, Any, Tuple, List


# в сценарии зашить бакеты со ставками

CONFIG = {
    "features": [
        "VTB_rate_[vip]_[0_15kk)",
        "VTB_rate_[vip]_[15kk_30kk)",
        "VTB_rate_[vip]_[30kk_50kk)",
        "VTB_rate_[vip]_[50kk_100kk)",
        "VTB_rate_[vip]_[100kk_200kk)",
        "VTB_rate_[vip]_[200kk_300kk)",
        "VTB_rate_[vip]_[300kk_500kk)",
        "VTB_rate_[vip]_[500kk_inf)",
    ],
    # тут важно пронать по порядку доли
    "balance_buckets": VIP_BALANCE_BUCKETS,
    # здесь задаем спреды
    "full_features": [
        "spread_VTB_rate_[vip]_[[15kk_30kk)-[0_15kk)]",
        "spread_VTB_rate_[vip]_[[30kk_50kk)-[15kk_30kk)]",
        "spread_VTB_rate_[vip]_[[50kk_100kk)-[30kk_50kk)]",
        "spread_VTB_rate_[vip]_[[100kk_200kk)-[50kk_100kk)]",
        "spread_VTB_rate_[vip]_[[200kk_300kk)-[100kk_200kk)]",
        "spread_VTB_rate_[vip]_[[300kk_500kk)-[200kk_300kk)]",
        "spread_VTB_rate_[vip]_[[500kk_inf)-[300kk_500kk)]",
    ],
    "target": VIP_BALANCE_BUCKETS,
    "estimator": calc_model_bucket_share(),
    "table_name": None,
    "table_date_col": _REPORT_DT_COLUMN,
    "default_start_date": datetime(2014, 2, 1),
    "model_name": f"newbusiness_vip_model_buckets",
}


class NewbusinessBucketsVipModel:
    def __init__(self, config=CONFIG):
        for key, value in config.items():
            setattr(self, key, value)

    def _generate_features(self, X):
        X_exog = pd.DataFrame(index=X.index)

        X_exog.loc[:, "spread_VTB_rate_[vip]_[[15kk_30kk)-[0_15kk)]"] = (
            X.loc[:, "VTB_rate_[vip]_[15kk_30kk)"] - X.loc[:, "VTB_rate_[vip]_[0_15kk)"]
        )

        X_exog.loc[:, "spread_VTB_rate_[vip]_[[30kk_50kk)-[15kk_30kk)]"] = (
            X.loc[:, "VTB_rate_[vip]_[30kk_50kk)"]
            - X.loc[:, "VTB_rate_[vip]_[15kk_30kk)"]
        )

        X_exog.loc[:, "spread_VTB_rate_[vip]_[[50kk_100kk)-[30kk_50kk)]"] = (
            X.loc[:, "VTB_rate_[vip]_[50kk_100kk)"]
            - X.loc[:, "VTB_rate_[vip]_[30kk_50kk)"]
        )

        X_exog.loc[:, "spread_VTB_rate_[vip]_[[100kk_200kk)-[50kk_100kk)]"] = (
            X.loc[:, "VTB_rate_[vip]_[100kk_200kk)"]
            - X.loc[:, "VTB_rate_[vip]_[50kk_100kk)"]
        )

        X_exog.loc[:, "spread_VTB_rate_[vip]_[[200kk_300kk)-[100kk_200kk)]"] = (
            X.loc[:, "VTB_rate_[vip]_[200kk_300kk)"]
            - X.loc[:, "VTB_rate_[vip]_[100kk_200kk)"]
        )

        X_exog.loc[:, "spread_VTB_rate_[vip]_[[300kk_500kk)-[200kk_300kk)]"] = (
            X.loc[:, "VTB_rate_[vip]_[300kk_500kk)"]
            - X.loc[:, "VTB_rate_[vip]_[200kk_300kk)"]
        )

        X_exog.loc[:, "spread_VTB_rate_[vip]_[[500kk_inf)-[300kk_500kk)]"] = (
            X.loc[:, "VTB_rate_[vip]_[500kk_inf)"]
            - X.loc[:, "VTB_rate_[vip]_[300kk_500kk)"]
        )

        X_exog = X_exog[self.full_features]

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

    # тут особо не обучаем
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        pass

    def predict(self, X: pd.DataFrame):
        verify_data(X, self.features)
        pass


#         y_hat = np.round(self.estimator.predict(X[self.full_features]),2)
#         y_hat = np.maximum(y_hat, self.min_model_predict)

#         # корректируем
#         y_hat = self._correct_pred_feature(X, y_hat).values

#         y_hat = pd.DataFrame(data = y_hat, columns = self.target, index = X.index)
#         y_hat.index.name = 'report_dt'
#         return y_hat


class NewbusinessBucketsVipDataLoader(SimpleDataLoader):
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

    ####################
    #     переопределим часть функций, т.к. мы не заполняем модель

    def get_maximum_train_range(self, spark) -> Tuple[datetime, datetime]:
        start_date = datetime(2010, 1, 31)
        end_date = None
        return (start_date, None)

    def get_training_data(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        return {"features": pd.DataFrame(), "target": pd.DataFrame()}

    def get_prediction_data(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        return {"features": pd.DataFrame()}

    def get_ground_truth(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        return {"target": pd.DataFrame()}

    def get_portfolio(
        self, spark, report_date: datetime, params: Dict[str, Any] = None
    ) -> Dict[str, pd.DataFrame]:
        report_month = "-".join(str(report_date.date()).split("-")[:2])
        print("report_month:", report_month)


class NewbusinessBucketsVipModelTrainer(SimpleModelTrainer):
    def __init__(self, config=CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = NewbusinessBucketsVipDataLoader()
        self.model = NewbusinessBucketsVipModel()


class NewbusinessBucketsVipModelAdapter(SimpleModelAdapter):
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
        # портфолио
        portfolio_dt = min(list(forecast_context.model_data["portfolio"].keys()))
        port = forecast_context.model_data["portfolio"][portfolio_dt].copy()

        # селектим необходимую дату для фич
        forecast_dates = forecast_context.forecast_dates

        # парсим доли с портфеля
        parse_res = parse_buckets_from_port(
            port, segment="vip", balance_buckets=self.balance_buckets
        )

        # читаем признаки для предикта
        try:
            df_date = forecast_context.model_data["features"].loc[
                forecast_dates, self.features
            ]

            # в случае пропусков заполняем нуллами
            df_date = df_date.fillna(0)

        except KeyError:
            print(
                "Не заданы поля для ставок в разрезе балансов открытий вкладов для ВИП сегмента. Модель доли бакетов будет работать константано."
            )
            forecast_context.model_data["features"].loc[
                forecast_dates, self.features
            ] = 0
            df_date = forecast_context.model_data["features"].loc[
                forecast_dates, self.features
            ]

        # генерируем признаки
        X_features = NewbusinessBucketsVipModel._generate_features(self, X=df_date)

        parse_res_new = calc_new_shares(
            parse_res, self.balance_buckets, "vip", X_features
        )

        pred = pd.DataFrame(data=parse_res_new, index=forecast_dates).add_prefix(
            "bucket_balance_share_[vip]_"
        )

        return pred


NewbusinessBucketsVip = ModelMetaInfo(
    model_name=CONFIG["model_name"],
    model_trainer=NewbusinessBucketsVipModelTrainer(),
    data_loader=NewbusinessBucketsVipDataLoader(),
    adapter=NewbusinessBucketsVipModelAdapter,
    segment="vip",
)
