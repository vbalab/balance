import logging
import pandas as pd

from os import path
from glob import glob
from io import BytesIO
from enum import auto, Enum
from dataclasses import dataclass
from abc import abstractmethod, ABC
from typing import Dict, List, Type, Any

from core.calculator.core import Settings
from core.upfm.commons import ModelInfo, BaseModel, Scenario, ForecastContext

logging.config.dictConfig(Settings.LOGGING_CONFIG)
logger = logging.getLogger("core")


class CalculationType(Enum):  # TODO: why do we need it at all
    pass


class CurrentAccountsCalculationType(CalculationType):  # TODO: why here??
    CurrentAccountsBalance = auto()


@dataclass
class CalculationResult:
    calc_type: CalculationType  # it is not even used
    config: Scenario  # TODO: rename to `scenario`, because we have `BaseConfig` in engine.py
    calculated_data: Dict[str, pd.DataFrame]


class ModelRegister:
    """
    Класс представляет собой интерфейс для добавления и хранения _уже обученных_ моделей.
    Как правило используется только в ходе расчетов внутри калькулятора.
    """

    MODEL_FILE_MASK = "*.*"

    def __init__(
        self,
        adapter_types: Dict[str, Type[BaseModel]] = None,
        skip_errors: bool = True,
    ) -> None:
        self._adapter_types: Dict[str, Type[BaseModel]] = adapter_types
        self._skip_errors: bool = skip_errors

        self._models: Dict[ModelInfo, BaseModel] = {}

    def _find_adapter_type(self, model_info_: ModelInfo):
        adapter_type_: Type[BaseModel] = self._adapter_types.get(model_info_.prefix)

        if not adapter_type_ and not self._skip_errors:
            raise RuntimeError(f"Unsupported model {model_info_.prefix}")

        return adapter_type_

    def add_models_from_folder(self, folder: str, recursive: bool = False) -> None:
        if not folder:
            return

        mask: str = path.join(folder, ModelRegister.MODEL_FILE_MASK)
        files = glob(mask, recursive=recursive)

        for file_ in files:
            model_info_: ModelInfo = ModelInfo.from_str(
                path.splitext(path.basename(file_))[0]
            )

            adapter_type: type[BaseModel] = self._find_adapter_type(model_info_)

            if adapter_type:
                model_ = adapter_type(model_info_, file_)
                self._models[model_info_] = model_

    def add_models_from_folders(
        self,
        folders: List[str],
        recursive: bool = False,
    ) -> None:
        for folder in folders:
            self.add_models_from_folder(folder, recursive=recursive)

    def add_models_from_bytes(self, model_data: Dict[ModelInfo, BytesIO]) -> None:
        logger.info("add_models_from_bytes")

        for model_info, data in model_data.items():
            adapter_type: type[BaseModel] = self._find_adapter_type(model_info)

            if not adapter_type:
                logger.warning(f"missing adapter for {model_info}")
                continue

            logger.info(f"{model_info} - adapter {adapter_type}")

            try:
                self._models[model_info] = adapter_type(model_info, data)
            except Exception as e:
                logger.exception(e)
                raise ValueError(
                    f"Could not create instance for {adapter_type} with {model_info}"
                )

    def get_model(self, model_info: ModelInfo) -> BaseModel:
        return self._models[model_info]

    def add_model(self, model: BaseModel) -> None:
        self._models[model.model_info] = model

    def add_models(self, models: List[BaseModel]) -> None:
        [self.add_model(model_) for model_ in models]

    def list_models(self):
        return self._models


class AbstractCalculator(ABC):
    """
    Калькулятор - это класс, в котором происходят все манипуляции
    для получения прогноза, например: добавление данных сценария,
    подгрузка дополнительных фичей, пересчет фичей, прогноз моделей,
    согласование прогнозов моделей, формирование конечного
    резултата пргноза и т. п.
    """

    def __init__(
        self,
        model_register: ModelRegister,
        models: Dict[str, ModelInfo],   # TODO: this is in ModelRegister already, because `str` here is ModelPrefix in `model_register.adapter_types[str]`
        # TODO: receive ForecastContext, not scenario & model_data
        scenario: Scenario,
        model_data: Dict[str, Any] = None,
    ) -> None:
        self._model_register: ModelRegister = model_register
        self._models = models
        self._scenario: Scenario = scenario
        self._forecast_context = ForecastContext(
            # TODO: you see how strange it is to have scenario data in ForecastContext with scenario itself
            scenario.portfolio_dt,
            scenario.horizon,
            scenario,
            model_data,
        )
        # TODO: this class has BaseModel & ForecastContext -> FullFitPredictModel

    @abstractmethod
    def calculate(self, calc_type: CalculationType) -> CalculationResult:
        pass


class SingleModelCalculator(AbstractCalculator):    # TODO: check if used at all
    # TODO: the same applies here
    def __init__(
        self,
        model_register: ModelRegister,
        models: Dict[str, ModelInfo],
        scenario: pd.DataFrame,
        model_data: Dict[str, Any] = None,
    ) -> None:
        super().__init__(model_register, models, scenario, model_data)

        self._model: BaseModel = model_register.get_model(
            self._models[next(iter(self._models))]
        )

    def calculate(self, calc_type: CalculationType) -> CalculationResult:
        df_predictions: pd.DataFrame = self._model.predict(self._forecast_context)

        return CalculationResult(
            calc_type,
            self._scenario,
            {calc_type.name: df_predictions},
        )
