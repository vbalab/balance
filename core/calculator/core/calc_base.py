"""Foundational calculator abstractions and shared utilities."""

from __future__ import annotations

import logging
import logging.config
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from glob import glob
from io import BytesIO
from os import path
from typing import Any, Dict, Iterable, Optional, Type

import pandas as pd  # type: ignore[import-untyped]

from core.calculator.core import Settings
from core.upfm.commons import ModelInfo, BaseModel, Scenario, ForecastContext

logging.config.dictConfig(Settings.LOGGING_CONFIG)
logger = logging.getLogger("core")


class CalculationType(Enum):  # TODO: why do we need it at all
    """Marker enumeration for calculator-specific calculation flavours."""

    pass


class CurrentAccountsCalculationType(CalculationType):  # TODO: why here??
    """Calculation types that are specific to current-account scenarios."""

    CurrentAccountsBalance = auto()


@dataclass
class CalculationResult:
    """Container describing the outcome of a calculator run."""

    calc_type: CalculationType  # it is not even used
    config: Scenario  # TODO: rename to `scenario`, because we have `BaseConfig` in engine.py
    calculated_data: Dict[str, pd.DataFrame]


class ModelRegister:
    """Registry that exposes access to pre-trained model instances."""

    MODEL_FILE_MASK = "*.*"

    def __init__(
        self,
        adapter_types: Optional[Dict[str, Type[BaseModel]]] = None,
        skip_errors: bool = True,
    ) -> None:
        self._adapter_types: Optional[Dict[str, Type[BaseModel]]] = adapter_types
        self._skip_errors: bool = skip_errors

        self._models: Dict[ModelInfo, BaseModel] = {}

    def _find_adapter_type(
        self, model_info_: ModelInfo
    ) -> Optional[Type[BaseModel]]:
        """Return a model adapter type for the provided :class:`ModelInfo`."""

        adapter_type_: Optional[Type[BaseModel]] = None
        if self._adapter_types is not None:
            adapter_type_ = self._adapter_types.get(model_info_.prefix)

        if not adapter_type_ and not self._skip_errors:
            raise RuntimeError(f"Unsupported model {model_info_.prefix}")

        return adapter_type_

    def add_models_from_folder(self, folder: str, recursive: bool = False) -> None:
        """Register models located inside *folder* according to adapter bindings."""

        if not folder:
            return

        mask: str = path.join(folder, ModelRegister.MODEL_FILE_MASK)
        files: Iterable[str] = glob(mask, recursive=recursive)

        for file_ in files:
            model_info_: ModelInfo = ModelInfo.from_str(
                path.splitext(path.basename(file_))[0]
            )

            adapter_type: Optional[Type[BaseModel]] = self._find_adapter_type(
                model_info_
            )

            if adapter_type:
                model_ = adapter_type(model_info_, file_)
                self._models[model_info_] = model_

    def add_models_from_folders(
        self, folders: Iterable[str], recursive: bool = False
    ) -> None:
        """Register models from every folder listed in *folders*."""

        for folder in folders:
            self.add_models_from_folder(folder, recursive=recursive)

    def add_models_from_bytes(self, model_data: Dict[ModelInfo, BytesIO]) -> None:
        """Register models from in-memory serialized blobs."""

        logger.info("add_models_from_bytes")

        for model_info, data in model_data.items():
            adapter_type: Optional[Type[BaseModel]] = self._find_adapter_type(model_info)

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
        """Return the instantiated model corresponding to *model_info*."""

        return self._models[model_info]

    def add_model(self, model: BaseModel) -> None:
        """Add a pre-instantiated *model* to the registry."""

        self._models[model.model_info] = model

    def add_models(self, models: Iterable[BaseModel]) -> None:
        """Bulk register each model from *models*."""

        for model_ in models:
            self.add_model(model_)

    def list_models(self) -> Dict[ModelInfo, BaseModel]:
        """Return every registered model keyed by its :class:`ModelInfo`."""

        return self._models


class AbstractCalculator(ABC):
    """Base class describing the workflow required to build a forecast."""

    def __init__(
        self,
        model_register: ModelRegister,
        models: Dict[str, ModelInfo],
        # TODO: receive ForecastContext, not scenario & model_data
        scenario: Scenario,
        model_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._model_register: ModelRegister = model_register
        self._models: Dict[str, ModelInfo] = models
        self._scenario: Scenario = scenario
        self._forecast_context = ForecastContext(
            scenario.portfolio_dt,
            scenario.horizon,
            scenario,
            model_data or {},
        )
        # TODO: this class has BaseModel & ForecastContext -> FullFitPredictModel

    @abstractmethod
    def calculate(self, calc_type: CalculationType) -> CalculationResult:
        """Execute the calculation for *calc_type* and return the result."""

        pass


class SingleModelCalculator(AbstractCalculator):  # TODO: check if used at all
    """Calculator that delegates the entire forecast to a single model."""

    # TODO: the same applies here
    def __init__(
        self,
        model_register: ModelRegister,
        models: Dict[str, ModelInfo],
        scenario: Scenario,
        model_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model_register, models, scenario, model_data)

        self._model: BaseModel = model_register.get_model(
            self._models[next(iter(self._models))]
        )

    def calculate(self, calc_type: CalculationType) -> CalculationResult:
        """Run the wrapped model and return its predictions as a result."""

        df_predictions: pd.DataFrame = self._model.predict(self._forecast_context)

        return CalculationResult(
            calc_type,
            self._scenario,
            {calc_type.name: df_predictions},
        )
