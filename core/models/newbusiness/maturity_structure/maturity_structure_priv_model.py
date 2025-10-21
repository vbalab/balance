from .maturity_structure_common_model import (
    MaturityStructureBaseModel,
    MaturityStructureBaseDataLoader,
    MaturityStructureBaseModelTrainer,
    MaturityStructureBaseModelAdapter,
    BASE_CONFIG,
    gen_maturity_model_name,
)
from datetime import datetime
from core.upfm.commons import (
    ModelInfo,
    ModelMetaInfo,
    ModelContainer,
)

R0S0_CONFIG = {
    "model_name": gen_maturity_model_name(segment="priv", repl=0, sub=0),
    "m": 1.037,
    "target": [
        "y_inflow_share_[priv]_[r0s0]_[90d]",
        "y_inflow_share_[priv]_[r0s0]_[180d]",
        "y_inflow_share_[priv]_[r0s0]_[365d]",
        "y_inflow_share_[priv]_[r0s0]_[548d]",
        "y_inflow_share_[priv]_[r0s0]_[730d]",
        "y_inflow_share_[priv]_[r0s0]_[1095d]",
    ],
    "features": [
        "VTB_weighted_rate_[priv]_[r0s0]_[90d]",
        "VTB_weighted_rate_[priv]_[r0s0]_[180d]",
        "VTB_weighted_rate_[priv]_[r0s0]_[365d]",
        "VTB_weighted_rate_[priv]_[r0s0]_[548d]",
        "VTB_weighted_rate_[priv]_[r0s0]_[730d]",
        "VTB_weighted_rate_[priv]_[r0s0]_[1095d]",
    ],
    "weight_months": BASE_CONFIG["weight_months"],
    "table_name": BASE_CONFIG["table_name"],
    "table_date_col": BASE_CONFIG["table_date_col"],
    "default_start_date": datetime(2014, 1, 1),
    "inflow_threshold": 1e9,
}

R0S1_CONFIG = {
    "model_name": gen_maturity_model_name(segment="priv", repl=0, sub=1),
    "m": 1.1,
    "target": [
        "y_inflow_share_[priv]_[r0s1]_[90d]",
        "y_inflow_share_[priv]_[r0s1]_[180d]",
        "y_inflow_share_[priv]_[r0s1]_[365d]",
        "y_inflow_share_[priv]_[r0s1]_[548d]",
        "y_inflow_share_[priv]_[r0s1]_[730d]",
        "y_inflow_share_[priv]_[r0s1]_[1095d]",
    ],
    "features": [
        "VTB_weighted_rate_[priv]_[r0s1]_[90d]",
        "VTB_weighted_rate_[priv]_[r0s1]_[180d]",
        "VTB_weighted_rate_[priv]_[r0s1]_[365d]",
        "VTB_weighted_rate_[priv]_[r0s1]_[548d]",
        "VTB_weighted_rate_[priv]_[r0s1]_[730d]",
        "VTB_weighted_rate_[priv]_[r0s1]_[1095d]",
    ],
    "weight_months": BASE_CONFIG["weight_months"],
    "table_name": BASE_CONFIG["table_name"],
    "table_date_col": BASE_CONFIG["table_date_col"],
    "default_start_date": datetime(2020, 1, 1),
    "inflow_threshold": None,
}

R1S0_CONFIG = {
    "model_name": gen_maturity_model_name(segment="priv", repl=1, sub=0),
    "m": 1.05,
    "target": [
        "y_inflow_share_[priv]_[r1s0]_[90d]",
        "y_inflow_share_[priv]_[r1s0]_[180d]",
        "y_inflow_share_[priv]_[r1s0]_[365d]",
        "y_inflow_share_[priv]_[r1s0]_[548d]",
        "y_inflow_share_[priv]_[r1s0]_[730d]",
        "y_inflow_share_[priv]_[r1s0]_[1095d]",
    ],
    "features": [
        "VTB_weighted_rate_[priv]_[r1s0]_[90d]",
        "VTB_weighted_rate_[priv]_[r1s0]_[180d]",
        "VTB_weighted_rate_[priv]_[r1s0]_[365d]",
        "VTB_weighted_rate_[priv]_[r1s0]_[548d]",
        "VTB_weighted_rate_[priv]_[r1s0]_[730d]",
        "VTB_weighted_rate_[priv]_[r1s0]_[1095d]",
    ],
    "weight_months": BASE_CONFIG["weight_months"],
    "table_name": BASE_CONFIG["table_name"],
    "table_date_col": BASE_CONFIG["table_date_col"],
    "default_start_date": datetime(2014, 1, 1),
    "inflow_threshold": 1e8,
}

R1S1_CONFIG = {
    "model_name": gen_maturity_model_name(segment="priv", repl=1, sub=1),
    "m": 1.0122,
    "target": [
        "y_inflow_share_[priv]_[r1s1]_[90d]",
        "y_inflow_share_[priv]_[r1s1]_[180d]",
        "y_inflow_share_[priv]_[r1s1]_[365d]",
        "y_inflow_share_[priv]_[r1s1]_[548d]",
        "y_inflow_share_[priv]_[r1s1]_[730d]",
        "y_inflow_share_[priv]_[r1s1]_[1095d]",
    ],
    "features": [
        "VTB_weighted_rate_[priv]_[r1s1]_[90d]",
        "VTB_weighted_rate_[priv]_[r1s1]_[180d]",
        "VTB_weighted_rate_[priv]_[r1s1]_[365d]",
        "VTB_weighted_rate_[priv]_[r1s1]_[548d]",
        "VTB_weighted_rate_[priv]_[r1s1]_[730d]",
        "VTB_weighted_rate_[priv]_[r1s1]_[1095d]",
    ],
    "weight_months": BASE_CONFIG["weight_months"],
    "table_name": BASE_CONFIG["table_name"],
    "table_date_col": BASE_CONFIG["table_date_col"],
    "default_start_date": datetime(2014, 1, 1),
    "inflow_threshold": 0.8e9,
}


# R0S0
# ______________________________________________________________________
class MaturityStructurePrivR0S0Model(MaturityStructureBaseModel):
    def __init__(self, config=R0S0_CONFIG):
        super().__init__(config)


class MaturityStructurePrivR0S0DataLoader(MaturityStructureBaseDataLoader):
    def __init__(self, config=R0S0_CONFIG):
        super().__init__(config)


class MaturityStructurePrivR0S0ModelTrainer(MaturityStructureBaseModelTrainer):
    def __init__(self, config=R0S0_CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = MaturityStructurePrivR0S0DataLoader()
        self.model = MaturityStructurePrivR0S0Model()


class MaturityStructurePrivR0S0ModelAdapter(MaturityStructureBaseModelAdapter):
    def __init__(
        self, model_info_: ModelInfo, filepath_or_buffer, config: dict = R0S0_CONFIG
    ):
        super().__init__(model_info_, filepath_or_buffer, config)


MaturityStructurePrivR0S0 = ModelMetaInfo(
    model_name=R0S0_CONFIG["model_name"],
    model_trainer=MaturityStructurePrivR0S0ModelTrainer(),
    data_loader=MaturityStructurePrivR0S0DataLoader(),
    adapter=MaturityStructurePrivR0S0ModelAdapter,
    segment="priv",
    replenishable_flg=0,
    subtraction_flg=0,
)
# __________________________________________________________________________


class MaturityStructurePrivR0S1Model(MaturityStructureBaseModel):
    def __init__(self, config=R0S1_CONFIG):
        super().__init__(config)


class MaturityStructurePrivR0S1DataLoader(MaturityStructureBaseDataLoader):
    def __init__(self, config=R0S1_CONFIG):
        super().__init__(config)


class MaturityStructurePrivR0S1ModelTrainer(MaturityStructureBaseModelTrainer):
    def __init__(self, config=R0S1_CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = MaturityStructurePrivR0S1DataLoader()
        self.model = MaturityStructurePrivR0S1Model()


class MaturityStructurePrivR0S1ModelAdapter(MaturityStructureBaseModelAdapter):
    def __init__(
        self, model_info_: ModelInfo, filepath_or_buffer, config: dict = R0S1_CONFIG
    ):
        super().__init__(model_info_, filepath_or_buffer, config)


MaturityStructurePrivR0S1 = ModelMetaInfo(
    model_name=R0S1_CONFIG["model_name"],
    model_trainer=MaturityStructurePrivR0S1ModelTrainer(),
    data_loader=MaturityStructurePrivR0S1DataLoader(),
    adapter=MaturityStructurePrivR0S1ModelAdapter,
    segment="priv",
    replenishable_flg=0,
    subtraction_flg=1,
)


# ______________________________________________________________________________
class MaturityStructurePrivR1S0Model(MaturityStructureBaseModel):
    def __init__(self, config=R1S0_CONFIG):
        super().__init__(config)


class MaturityStructurePrivR1S0DataLoader(MaturityStructureBaseDataLoader):
    def __init__(self, config=R1S0_CONFIG):
        super().__init__(config)


class MaturityStructurePrivR1S0ModelTrainer(MaturityStructureBaseModelTrainer):
    def __init__(self, config=R1S0_CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = MaturityStructurePrivR1S0DataLoader()
        self.model = MaturityStructurePrivR1S0Model()


class MaturityStructurePrivR1S0ModelAdapter(MaturityStructureBaseModelAdapter):
    def __init__(
        self, model_info_: ModelInfo, filepath_or_buffer, config: dict = R1S0_CONFIG
    ):
        super().__init__(model_info_, filepath_or_buffer, config)


MaturityStructurePrivR1S0 = ModelMetaInfo(
    model_name=R1S0_CONFIG["model_name"],
    model_trainer=MaturityStructurePrivR1S0ModelTrainer(),
    data_loader=MaturityStructurePrivR1S0DataLoader(),
    adapter=MaturityStructurePrivR1S0ModelAdapter,
    segment="priv",
    replenishable_flg=1,
    subtraction_flg=0,
)


# ______________________________________________________________________________
class MaturityStructurePrivR1S1Model(MaturityStructureBaseModel):
    def __init__(self, config=R1S1_CONFIG):
        super().__init__(config)


class MaturityStructurePrivR1S1DataLoader(MaturityStructureBaseDataLoader):
    def __init__(self, config=R1S1_CONFIG):
        super().__init__(config)


class MaturityStructurePrivR1S1ModelTrainer(MaturityStructureBaseModelTrainer):
    def __init__(self, config=R1S1_CONFIG) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.dataloader = MaturityStructurePrivR1S1DataLoader()
        self.model = MaturityStructurePrivR1S1Model()


class MaturityStructurePrivR1S1ModelAdapter(MaturityStructureBaseModelAdapter):
    def __init__(
        self, model_info_: ModelInfo, filepath_or_buffer, config: dict = R1S1_CONFIG
    ):
        super().__init__(model_info_, filepath_or_buffer, config)


MaturityStructurePrivR1S1 = ModelMetaInfo(
    model_name=R1S1_CONFIG["model_name"],
    model_trainer=MaturityStructurePrivR1S1ModelTrainer(),
    data_loader=MaturityStructurePrivR1S1DataLoader(),
    adapter=MaturityStructurePrivR1S1ModelAdapter,
    segment="priv",
    replenishable_flg=1,
    subtraction_flg=1,
)

MaturityStructurePriv = ModelContainer(
    models=(
        MaturityStructurePrivR0S0,
        MaturityStructurePrivR0S1,
        MaturityStructurePrivR1S0,
        MaturityStructurePrivR1S1,
    )
)
