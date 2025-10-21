from .maturity_structure_common_model import (
    MaturityStructureBaseModel,
    MaturityStructureBaseDataLoader,
    MaturityStructureBaseModelTrainer,
    MaturityStructureBaseModelAdapter,
    BASE_CONFIG,
    gen_maturity_model_name,
)
import pandas as pd
import numpy as np
from datetime import datetime
from core.upfm.commons import (
    DataLoader,
    BaseModel,
    ModelInfo,
    ForecastContext,
    _REPORT_DT_COLUMN,
    ModelMetaInfo,
    ModelContainer,
)

R0S0_CONFIG = {
    "model_name": gen_maturity_model_name(segment="vip", repl=0, sub=0),
    "m": 1.0425,
    "target": [
        "y_inflow_share_[vip]_[r0s0]_[90d]",
        "y_inflow_share_[vip]_[r0s0]_[180d]",
        "y_inflow_share_[vip]_[r0s0]_[365d]",
        "y_inflow_share_[vip]_[r0s0]_[548d]",
        "y_inflow_share_[vip]_[r0s0]_[730d]",
        "y_inflow_share_[vip]_[r0s0]_[1095d]",
    ],
    "features": [
        "VTB_weighted_rate_[vip]_[r0s0]_[90d]",
        "VTB_weighted_rate_[vip]_[r0s0]_[180d]",
        "VTB_weighted_rate_[vip]_[r0s0]_[365d]",
        "VTB_weighted_rate_[vip]_[r0s0]_[548d]",
        "VTB_weighted_rate_[vip]_[r0s0]_[730d]",
        "VTB_weighted_rate_[vip]_[r0s0]_[1095d]",
    ],
    "weight_months": BASE_CONFIG["weight_months"],
    "table_name": BASE_CONFIG["table_name"],
    "table_date_col": BASE_CONFIG["table_date_col"],
    "default_start_date": datetime(2014, 1, 1),
    "inflow_threshold": 1e9,
}

R0S1_CONFIG = {
    "model_name": gen_maturity_model_name(segment="vip", repl=0, sub=1),
    "m": 1.1,
    "target": [
        "y_inflow_share_[vip]_[r0s1]_[90d]",
        "y_inflow_share_[vip]_[r0s1]_[180d]",
        "y_inflow_share_[vip]_[r0s1]_[365d]",
        "y_inflow_share_[vip]_[r0s1]_[548d]",
        "y_inflow_share_[vip]_[r0s1]_[730d]",
        "y_inflow_share_[vip]_[r0s1]_[1095d]",
    ],
    "features": [
        "VTB_weighted_rate_[vip]_[r0s1]_[90d]",
        "VTB_weighted_rate_[vip]_[r0s1]_[180d]",
        "VTB_weighted_rate_[vip]_[r0s1]_[365d]",
        "VTB_weighted_rate_[vip]_[r0s1]_[548d]",
        "VTB_weighted_rate_[vip]_[r0s1]_[730d]",
        "VTB_weighted_rate_[vip]_[r0s1]_[1095d]",
    ],
    "weight_months": BASE_CONFIG["weight_months"],
    "table_name": BASE_CONFIG["table_name"],
    "table_date_col": BASE_CONFIG["table_date_col"],
    "default_start_date": datetime(2020, 1, 1),
    "inflow_threshold": None,
}

R1S0_CONFIG = {
    "model_name": gen_maturity_model_name(segment="vip", repl=1, sub=0),
    "m": 1.05,
    "target": [
        "y_inflow_share_[vip]_[r1s0]_[90d]",
        "y_inflow_share_[vip]_[r1s0]_[180d]",
        "y_inflow_share_[vip]_[r1s0]_[365d]",
        "y_inflow_share_[vip]_[r1s0]_[548d]",
        "y_inflow_share_[vip]_[r1s0]_[730d]",
        "y_inflow_share_[vip]_[r1s0]_[1095d]",
    ],
    "features": [
        "VTB_weighted_rate_[vip]_[r1s0]_[90d]",
        "VTB_weighted_rate_[vip]_[r1s0]_[180d]",
        "VTB_weighted_rate_[vip]_[r1s0]_[365d]",
        "VTB_weighted_rate_[vip]_[r1s0]_[548d]",
        "VTB_weighted_rate_[vip]_[r1s0]_[730d]",
        "VTB_weighted_rate_[vip]_[r1s0]_[1095d]",
    ],
    "weight_months": BASE_CONFIG["weight_months"],
    "table_name": BASE_CONFIG["table_name"],
    "table_date_col": BASE_CONFIG["table_date_col"],
    "default_start_date": datetime(2014, 1, 1),
    "inflow_threshold": 1e8,
}

# TODO: настроить гиперпараметр m
R1S1_CONFIG = {
    "model_name": gen_maturity_model_name(segment="vip", repl=1, sub=1),
    "m": 1.0465,
    "target": [
        "y_inflow_share_[vip]_[r1s1]_[90d]",
        "y_inflow_share_[vip]_[r1s1]_[180d]",
        "y_inflow_share_[vip]_[r1s1]_[365d]",
        "y_inflow_share_[vip]_[r1s1]_[548d]",
        "y_inflow_share_[vip]_[r1s1]_[730d]",
        "y_inflow_share_[vip]_[r1s1]_[1095d]",
    ],
    "features": [
        "VTB_weighted_rate_[vip]_[r1s1]_[90d]",
        "VTB_weighted_rate_[vip]_[r1s1]_[180d]",
        "VTB_weighted_rate_[vip]_[r1s1]_[365d]",
        "VTB_weighted_rate_[vip]_[r1s1]_[548d]",
        "VTB_weighted_rate_[vip]_[r1s1]_[730d]",
        "VTB_weighted_rate_[vip]_[r1s1]_[1095d]",
    ],
    "weight_months": BASE_CONFIG["weight_months"],
    "table_name": BASE_CONFIG["table_name"],
    "table_date_col": BASE_CONFIG["table_date_col"],
    "default_start_date": datetime(2014, 1, 1),
    "inflow_threshold": 1e9,
}


class MaturityStructureVipR0S0Model(MaturityStructureBaseModel):
    def __init__(self, config=R0S0_CONFIG):
        super().__init__(config)


class MaturityStructureVipR0S0DataLoader(MaturityStructureBaseDataLoader):
    def __init__(self, config=R0S0_CONFIG):
        super().__init__(config)


class MaturityStructureVipR0S0ModelTrainer(MaturityStructureBaseModelTrainer):
    def __init__(self, config=R0S0_CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = MaturityStructureVipR0S0DataLoader()
        self.model = MaturityStructureVipR0S0Model()


class MaturityStructureVipR0S0ModelAdapter(MaturityStructureBaseModelAdapter):
    def __init__(
        self, model_info_: ModelInfo, filepath_or_buffer, config: dict = R0S0_CONFIG
    ):
        super().__init__(model_info_, filepath_or_buffer, config)


MaturityStructureVipR0S0 = ModelMetaInfo(
    model_name=R0S0_CONFIG["model_name"],
    model_trainer=MaturityStructureVipR0S0ModelTrainer(),
    data_loader=MaturityStructureVipR0S0DataLoader(),
    adapter=MaturityStructureVipR0S0ModelAdapter,
    segment="vip",
    replenishable_flg=0,
    subtraction_flg=0,
)

# __________________________________________________________________________


class MaturityStructureVipR0S1Model(MaturityStructureBaseModel):
    def __init__(self, config=R0S1_CONFIG):
        super().__init__(config)


class MaturityStructureVipR0S1DataLoader(MaturityStructureBaseDataLoader):
    def __init__(self, config=R0S1_CONFIG):
        super().__init__(config)


class MaturityStructureVipR0S1ModelTrainer(MaturityStructureBaseModelTrainer):
    def __init__(self, config=R0S1_CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = MaturityStructureVipR0S1DataLoader()
        self.model = MaturityStructureVipR0S1Model()


class MaturityStructureVipR0S1ModelAdapter(MaturityStructureBaseModelAdapter):
    def __init__(
        self, model_info_: ModelInfo, filepath_or_buffer, config: dict = R0S1_CONFIG
    ):
        super().__init__(model_info_, filepath_or_buffer, config)


MaturityStructureVipR0S1 = ModelMetaInfo(
    model_name=R0S1_CONFIG["model_name"],
    model_trainer=MaturityStructureVipR0S1ModelTrainer(),
    data_loader=MaturityStructureVipR0S1DataLoader(),
    adapter=MaturityStructureVipR0S1ModelAdapter,
    segment="vip",
    replenishable_flg=0,
    subtraction_flg=1,
)


# ______________________________________________________________________________
class MaturityStructureVipR1S0Model(MaturityStructureBaseModel):
    def __init__(self, config=R1S0_CONFIG):
        super().__init__(config)


class MaturityStructureVipR1S0DataLoader(MaturityStructureBaseDataLoader):
    def __init__(self, config=R1S0_CONFIG):
        super().__init__(config)


class MaturityStructureVipR1S0ModelTrainer(MaturityStructureBaseModelTrainer):
    def __init__(self, config=R1S0_CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = MaturityStructureVipR1S0DataLoader()
        self.model = MaturityStructureVipR1S0Model()


class MaturityStructureVipR1S0ModelAdapter(MaturityStructureBaseModelAdapter):
    def __init__(
        self, model_info_: ModelInfo, filepath_or_buffer, config: dict = R1S0_CONFIG
    ):
        super().__init__(model_info_, filepath_or_buffer, config)


MaturityStructureVipR1S0 = ModelMetaInfo(
    model_name=R1S0_CONFIG["model_name"],
    model_trainer=MaturityStructureVipR1S0ModelTrainer(),
    data_loader=MaturityStructureVipR1S0DataLoader(),
    adapter=MaturityStructureVipR1S0ModelAdapter,
    segment="vip",
    replenishable_flg=1,
    subtraction_flg=0,
)


# ______________________________________________________________________________
class MaturityStructureVipR1S1Model(MaturityStructureBaseModel):
    def __init__(self, config=R1S1_CONFIG):
        super().__init__(config)


class MaturityStructureVipR1S1DataLoader(MaturityStructureBaseDataLoader):
    def __init__(self, config=R1S1_CONFIG):
        super().__init__(config)


class MaturityStructureVipR1S1ModelTrainer(MaturityStructureBaseModelTrainer):
    def __init__(self, config=R1S1_CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = MaturityStructureVipR1S1DataLoader()
        self.model = MaturityStructureVipR1S1Model()


class MaturityStructureVipR1S1ModelAdapter(MaturityStructureBaseModelAdapter):
    def __init__(
        self, model_info_: ModelInfo, filepath_or_buffer, config: dict = R1S1_CONFIG
    ):
        super().__init__(model_info_, filepath_or_buffer, config)


MaturityStructureVipR1S1 = ModelMetaInfo(
    model_name=R1S1_CONFIG["model_name"],
    model_trainer=MaturityStructureVipR1S1ModelTrainer(),
    data_loader=MaturityStructureVipR1S1DataLoader(),
    adapter=MaturityStructureVipR1S1ModelAdapter,
    segment="vip",
    replenishable_flg=1,
    subtraction_flg=1,
)

MaturityStructureVip = ModelContainer(
    models=(
        MaturityStructureVipR0S0,
        MaturityStructureVipR0S1,
        MaturityStructureVipR1S0,
        MaturityStructureVipR1S1,
    )
)
