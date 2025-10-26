from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from core.calculator.storage import ModelDB
from core.upfm.commons import DataLoader, MLException, ModelInfo, Scenario
from core.calculator.core import (
    AbstractCalculator,
    AbstractEngine,
    BaseConfig,
    CalculationResult,
    Settings,
)

logging.config.dictConfig(Settings.LOGGING_CONFIG)
logger = logging.getLogger("core")


@dataclass
class ForecastConfig(BaseConfig):
    # TODO: why not scenario
    scenario_data: Optional[pd.DataFrame] = None
    portfolio: Optional[pd.DataFrame] = None

    @property
    def portfolio_dt(self) -> datetime:
        return self.first_train_end_dt

    @property
    def train_ends(self) -> List[datetime]:
        return [self.first_train_end_dt]

    @property
    def forecast_dates(self) -> List[datetime]:
        return list(self._forecast_dates().values())[0]


class ForecastEngine(AbstractEngine):
    STEP_KEY = 1

    @property
    def calc_results(self) -> Dict[str, pd.DataFrame]:
        return self._calc_results[STEP_KEY].calculated_data

    def _create_calc(self) -> AbstractCalculator:
        dt_: datetime = self._config.train_ends[0]

        models: Dict[str, ModelInfo] = {
            tag: self.trained_models[(STEP_KEY, tag)] for tag in self._config.trainers
        }

        scenario_: Scenario = Scenario(
            dt_,
            self._config.horizon,
            self._config.scenario_data,
        )
        model_data: Dict[str, Any] = {
            "portfolio": {dt_: self._config.portfolio},
            "features": pd.DataFrame().rename_axis("report_dt"),
        }

        return self._config.calculator_type(
            self.register,
            models,
            scenario_,
            model_data,
        )

    def run_all(self) -> None:
        n_models = len(
            self._training_manager._db.find_trained_model_by_dt1(
                end_dt=self._config.first_train_end_dt
            )
        )

        if (n_models < 42) & (self._training_manager._spark is None):  # TODO: `42` ???
            raise MLException(
                f"Модель не поддерживает временной период: {self._config.train_ends[0]}"
            )

        self.train_models()

        logger.info("training models completed")

        self.calc: AbstractCalculator = self._create_calc()

        logger.info("calculation started")
        self._calc_results[STEP_KEY] = self.calc.calculate(self._config.calc_type)

    def save_to_db(
        self,
        model_db: ModelDB,
        analyzers: List[Any],
        segment_map: Dict[str, str],
    ) -> None:
        pass


class ForecastEngine1(AbstractEngine):  # TODO: delete?
    STEP_KEY = 1

    @property
    def calc_results(self) -> CalculationResult:
        return self._calc_results[ForecastEngine.STEP_KEY]

    def _load_data(self, tag: str) -> None:
        loader_: DataLoader = self._config.data_loaders[tag]
        logger.info(f"loading data for {tag}")
        self._portfolio_data[(ForecastEngine.STEP_KEY, tag)] = loader_.get_portfolio(
            self._spark, self._config.train_ends[0]
        )

    def _create_calc(self) -> AbstractCalculator:
        dt_: datetime = self._config.train_ends[0]
        models: Dict[str, ModelInfo] = {
            tag: self.trained_models[(ForecastEngine.STEP_KEY, tag)]
            for tag in self._config.trainers
        }
        scenario_: Scenario = Scenario(
            dt_, self._config.horizon, self._config.scenario_data
        )
        model_data: Dict[str, Any] = {
            f"{tag}_portfolio": self._portfolio_data[(ForecastEngine.STEP_KEY, tag)]
            for tag in self._config.data_loaders
        }
        return self._config.calculator_type(
            self.register, models, scenario_, model_data
        )

    def run_all(self) -> None:
        self.train_models()
        logger.info("training models completed")
        for tag in self._config.data_loaders:
            self._load_data(tag)
        logger.info("portfolio loading completed")
        calc: AbstractCalculator = self._create_calc()
        logger.info("calculation started")
        self._calc_results[ForecastEngine.STEP_KEY] = calc.calculate(
            self._config.calc_type
        )
