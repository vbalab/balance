from __future__ import annotations

"""Engines dedicated to running single-step forecast calculations."""

import logging
import logging.config
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd  # type: ignore[import-untyped]

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
    """Configuration values required to produce a single forecast run."""

    # TODO: why not scenario
    scenario_data: Optional[pd.DataFrame] = None
    portfolio: Optional[pd.DataFrame] = None

    @property
    def portfolio_dt(self) -> datetime:
        """Convenience accessor for the portfolio snapshot date."""

        if self.first_train_end_dt is None:
            raise ValueError("first_train_end_dt must be provided for forecasts")

        return self.first_train_end_dt

    @property
    def train_ends(self) -> List[datetime]:
        """Return the training end dates used by the forecast."""

        return [self.portfolio_dt]

    @property
    def forecast_dates(self) -> Dict[int, List[datetime]]:
        """Return the forecast horizon associated with this run."""

        return self._forecast_dates()

    @property
    def horizon_dates(self) -> List[datetime]:
        """Convenience accessor returning the single-step forecast horizon."""

        return list(self.forecast_dates.values())[0]


class ForecastEngine(AbstractEngine[ForecastConfig]):
    """Engine responsible for orchestrating a single forecast step."""

    STEP_KEY = 1

    @property
    def calculated_frames(self) -> Dict[str, pd.DataFrame]:
        """Expose the calculated data frame dictionary."""

        return self._calc_results[self.STEP_KEY].calculated_data

    def _create_calc(self) -> AbstractCalculator:
        """Instantiate the configured calculator for the forecast."""

        dt_: datetime = self._config.train_ends[0]

        models: Dict[str, ModelInfo] = {
            tag: self.trained_models[(self.STEP_KEY, tag)]
            for tag in self._config.trainers
        }

        scenario_data = self._config.scenario_data
        if scenario_data is None:
            raise ValueError("scenario_data must be provided for forecast execution")

        portfolio = self._config.portfolio
        if portfolio is None:
            raise ValueError("portfolio must be provided for forecast execution")

        scenario_: Scenario = Scenario(
            dt_,
            self._config.horizon,
            scenario_data,
        )
        model_data: Dict[str, Any] = {
            "portfolio": {dt_: portfolio},
            "features": pd.DataFrame().rename_axis("report_dt"),
        }

        calculator_cls = self._config.calculator_type
        if calculator_cls is None:
            raise ValueError("calculator_type must be provided for forecast execution")

        return calculator_cls(
            self.register,
            models,
            scenario_,
            model_data,
        )

    def run_all(self) -> None:
        """Train required models and execute the calculator once."""

        if self._config.first_train_end_dt is None:
            raise ValueError("first_train_end_dt must be provided for forecasts")

        model_db = self._training_manager._db
        if model_db is None:
            raise ValueError("Training manager is not configured with a model database")

        n_models = len(
            model_db.find_trained_model_by_dt1(end_dt=self._config.first_train_end_dt)
        )

        if (n_models < 42) and (
            self._training_manager._spark is None
        ):  # TODO: `42` ???
            raise MLException(
                f"Модель не поддерживает временной период: {self._config.train_ends[0]}"
            )

        self.train_models()

        logger.info("training models completed")

        self.calc: AbstractCalculator = self._create_calc()

        logger.info("calculation started")
        if self._config.calc_type is None:
            raise ValueError("calc_type must be provided for forecast execution")

        self._calc_results[self.STEP_KEY] = self.calc.calculate(self._config.calc_type)

    def save_to_db(
        self,
        model_db: ModelDB,
        analyzers: List[Any],
        segment_map: Dict[str, str],
    ) -> None:
        """Persist forecast results. Currently a stub implementation."""

        pass


class ForecastEngine1(AbstractEngine[ForecastConfig]):  # TODO: delete?
    """Legacy forecast engine variant that loads portfolio data explicitly."""

    STEP_KEY = 1

    @property
    def calculation_result(self) -> CalculationResult:
        """Return the raw :class:`CalculationResult` object."""

        return self._calc_results[ForecastEngine.STEP_KEY]

    def _load_data(self, tag: str) -> None:
        """Load and cache additional data required by the calculator."""

        loader_: DataLoader = self._config.data_loaders[tag]
        logger.info(f"loading data for {tag}")
        self._portfolio_data[(ForecastEngine.STEP_KEY, tag)] = loader_.get_portfolio(
            self._spark, self._config.train_ends[0]
        )

    def _create_calc(self) -> AbstractCalculator:
        """Instantiate the configured calculator for the legacy engine."""

        dt_: datetime = self._config.train_ends[0]
        models: Dict[str, ModelInfo] = {
            tag: self.trained_models[(ForecastEngine.STEP_KEY, tag)]
            for tag in self._config.trainers
        }
        scenario_data = self._config.scenario_data
        if scenario_data is None:
            raise ValueError("scenario_data must be provided for forecast execution")

        scenario_: Scenario = Scenario(dt_, self._config.horizon, scenario_data)
        model_data: Dict[str, Any] = {
            f"{tag}_portfolio": self._portfolio_data[(ForecastEngine.STEP_KEY, tag)]
            for tag in self._config.data_loaders
        }
        calculator_cls = self._config.calculator_type
        if calculator_cls is None:
            raise ValueError("calculator_type must be provided for forecast execution")

        return calculator_cls(self.register, models, scenario_, model_data)

    def run_all(self) -> None:
        """Run the legacy workflow, including portfolio loading."""

        self.train_models()
        logger.info("training models completed")
        for tag in self._config.data_loaders:
            self._load_data(tag)
        logger.info("portfolio loading completed")
        calc: AbstractCalculator = self._create_calc()
        logger.info("calculation started")
        if self._config.calc_type is None:
            raise ValueError("calc_type must be provided for forecast execution")

        self._calc_results[ForecastEngine.STEP_KEY] = calc.calculate(
            self._config.calc_type
        )
