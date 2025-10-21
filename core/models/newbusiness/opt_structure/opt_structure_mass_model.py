import pandas as pd
import numpy as np
from copy import deepcopy
from core.models.newbusiness.simple_adapters import (
    SimpleDataLoader,
    SimpleModelTrainer,
    SimpleModelAdapter,
)
from core.models.utils import gen_opt_model_name
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
    "features": None,
    "target": [
        "y_inflow_share_[mass]_[r0s0]",
        "y_inflow_share_[mass]_[r0s1]",
        "y_inflow_share_[mass]_[r1s0]",
        "y_inflow_share_[mass]_[r1s1]",
    ],
    "estimator": None,
    "months_rolling": 3,
    "table_name": "prod_dadm_alm_sbx.almde_fl_dpst_opt_structure_model_features",
    "table_date_col": _REPORT_DT_COLUMN,
    "default_start_date": datetime(2014, 1, 1),
    "model_name": gen_opt_model_name(segment="mass"),
}


class OptStructureMassModel:
    def __init__(self, config=CONFIG):
        for key, value in config.items():
            setattr(self, key, value)

    def fit(self, X, y):
        y = y[self.target]
        self.last_row_y = y.iloc[-self.months_rolling :, :]

    def predict(self, X=None):
        y_hat_list = []
        for step in range(X.shape[0]):
            y_hat_step = pd.DataFrame(
                data=self.last_row_y.iloc[-3:, :].mean(axis=0).values.reshape(1, -1),
                columns=self.target,
                index=[X.index[step]],
            )
            self.last_row_y = self.last_row_y.append(y_hat_step)
            y_hat_list.append(y_hat_step)
        y_hat = pd.concat(y_hat_list)
        return y_hat


class OptStructureMassDataLoader(SimpleDataLoader):
    def __init__(self, config=CONFIG):
        super().__init__(config)


class OptStructureMassModelTrainer(SimpleModelTrainer):
    def __init__(self, config=CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = OptStructureMassDataLoader()
        self.model = OptStructureMassModel()


class OptStructureMassModelAdapter(SimpleModelAdapter):
    def __init__(
        self, model_info_: ModelInfo, filepath_or_buffer, config: dict = CONFIG
    ):
        super().__init__(model_info_, filepath_or_buffer, config)


OptStructureMass = ModelMetaInfo(
    model_name=CONFIG["model_name"],
    model_trainer=OptStructureMassModelTrainer(),
    data_loader=OptStructureMassDataLoader(),
    adapter=OptStructureMassModelAdapter,
    segment="mass",
)
