import pandas as pd
import numpy as np
from copy import deepcopy
from core.models.newbusiness.simple_adapters import (
    SimpleDataLoader,
    SimpleModelTrainer,
    SimpleModelAdapter,
)
from core.models.utils import gen_sa_product_structure_model_name
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
from datetime import datetime
from typing import Dict, Any, Tuple, List


CONFIG = {
    "features": [
        "SA_avg_weighted_rate_[priv]_[classic]",
        "SA_avg_weighted_rate_[priv]_[kopilka]",
    ],
    "full_features": ["spread_[classic-kopilka]"],
    "target": [
        "SA_avg_balance_share_[priv]_[classic]",
        "SA_avg_balance_share_[priv]_[kopilka]",
    ],
    "table_name": "prod_dadm_alm_sbx.almde_fl_dpst_saving_accounts_monthly_feature",
    "table_date_col": "report_date",
    "default_start_date": datetime(2019, 1, 1),
    "model_name": gen_sa_product_structure_model_name(segment="priv"),
    "share_elastic": 3.73,
    "upper_bound": 0.88,
    "lower_bound": 0.1,
}


class ProductStructurePrivModel:
    def __init__(self, config=CONFIG):
        for key, value in config.items():
            setattr(self, key, value)

    def _generate_features(self, X):
        X_exog = pd.DataFrame(index=X.index)
        X_exog.loc[:, "spread_[classic-kopilka]"] = (
            X.loc[:, "SA_avg_weighted_rate_[priv]_[classic]"]
            - X.loc[:, "SA_avg_weighted_rate_[priv]_[kopilka]"]
        )
        X_exog = X_exog[self.full_features]

        return X_exog

    def predict(self, X: pd.DataFrame):
        sigmoid = lambda x: self.lower_bound + (
            1 / (1 + np.exp(-self.share_elastic * (x)))
        ) * (self.upper_bound - self.lower_bound)
        classic_share = sigmoid(self._generate_features(X))
        kopilka_share = 1 - classic_share
        y_hat = pd.concat([classic_share, kopilka_share], axis=1)
        y_hat.columns = self.target
        y_hat.index.name = "report_dt"

        return y_hat


class ProductStructurePrivDataLoader(SimpleDataLoader):
    def __init__(self, config=CONFIG):
        super().__init__(config)


class ProductStructurePrivModelTrainer(SimpleModelTrainer):
    def __init__(self, config=CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = ProductStructurePrivDataLoader()
        self.model = ProductStructurePrivModel()


class ProductStructurePrivModelAdapter(SimpleModelAdapter):
    def __init__(
        self, model_info_: ModelInfo, filepath_or_buffer, config: dict = CONFIG
    ):
        super().__init__(model_info_, filepath_or_buffer, config)


ProductStructurePriv = ModelMetaInfo(
    model_name=CONFIG["model_name"],
    model_trainer=ProductStructurePrivModelTrainer(),
    data_loader=ProductStructurePrivDataLoader(),
    adapter=ProductStructurePrivModelAdapter,
    segment="priv",
)
