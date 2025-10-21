from core.models.utils import gen_opt_model_name
import pandas as pd
import numpy as np
from core.upfm.commons import (
    DataLoader,
    BaseModel,
    _REPORT_DT_COLUMN,
    ModelInfo,
    ForecastContext,
    ModelMetaInfo,
    ModelContainer,
)
from core.models.newbusiness.simple_adapters import (
    SimpleDataLoader,
    SimpleModelTrainer,
    SimpleModelAdapter,
)
from datetime import datetime
from typing import Dict, Any

CONFIG = {
    "target": [
        "y_inflow_share_[bvip]_[r0s0]",
        "y_inflow_share_[bvip]_[r0s1]",
        "y_inflow_share_[bvip]_[r1s0]",
        "y_inflow_share_[bvip]_[r1s1]",
    ],
    "features": None,
    "table_name": "prod_dadm_alm_sbx.almde_fl_dpst_opt_structure_model_features",
    "table_date_col": _REPORT_DT_COLUMN,
    "default_start_date": datetime(2014, 1, 1),
    "model_name": gen_opt_model_name(segment="bvip"),
}


class OptStructureBvipModel:
    def __init__(self, config=CONFIG):
        for key, value in config.items():
            setattr(self, key, value)

    def fit(self, X, y):
        pass

    def predict(self, X):
        y_hat = pd.DataFrame(
            data=np.repeat([[0, 0, 0, 1]], X.shape[0], axis=0),
            columns=self.target,
            index=X.index,
        )
        return y_hat


class OptStructureBvipDataLoader(SimpleDataLoader):
    def __init__(self, config=CONFIG):
        super().__init__(config)

    def get_training_data(
        self,
        spark,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, pd.DataFrame]:
        return {"features": None, "target": None}


class OptStructureBvipModelTrainer(SimpleModelTrainer):
    def __init__(self, config=CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = OptStructureBvipDataLoader()
        self.model = OptStructureBvipModel()


class OptStructureBvipModelAdapter(SimpleModelAdapter):
    def __init__(
        self, model_info_: ModelInfo, filepath_or_buffer, config: dict = CONFIG
    ):
        super().__init__(model_info_, filepath_or_buffer, config)


OptStructureBvip = ModelMetaInfo(
    model_name=CONFIG["model_name"],
    model_trainer=OptStructureBvipModelTrainer(),
    data_loader=OptStructureBvipDataLoader(),
    adapter=OptStructureBvipModelAdapter,
    segment="bvip",
)
