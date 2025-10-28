"""Backtesting support for calculator pipelines."""

from __future__ import annotations

import logging
import logging.config
from dataclasses import dataclass
from datetime import datetime
from functools import reduce
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd  # type: ignore[import-untyped]
from pyspark.sql import SparkSession  # type: ignore[import-not-found]

from core.upfm.commons import (
    DataLoader,
    ModelInfo,
    Scenario,
    _REPORT_DT_COLUMN,
)
from core.calculator.core import (
    AbstractCalculator,
    AbstractEngine,
    BaseConfig,
    Settings,
    TrainingManager,
)
from core.calculator.storage import ModelDB, BackTestDataEntity, BackTestInfoEntity

logging.config.dictConfig(Settings.LOGGING_CONFIG)
logger = logging.getLogger("core")


@dataclass
class BackTestConfig(BaseConfig):
    """Configuration defining a rolling backtest."""

    steps: int = 1

    @property
    def train_ends(self) -> List[datetime]:
        """Return the end dates for each training iteration."""

        return self._train_ends()

    @property
    def forecast_dates(self) -> Dict[int, List[datetime]]:
        """Map backtest step indices to their forecast horizons."""

        return self._forecast_dates(self.steps)


class BackTestEngine(AbstractEngine[BackTestConfig]):
    """Engine that orchestrates multi-step backtesting workflows."""

    # TODO: better move to definitions.py
    DEPOSIT_SEGMENT_MAP: Dict[str, str] = {
        "_portfolio": "portfolio",
        "_novip_": "novip",
        "_vip_": "vip",
    }

    def __init__(
        self,
        spark: SparkSession,
        config: BackTestConfig,
        training_manager: TrainingManager,
        overwrite_models: bool = True,
    ) -> None:
        super().__init__(spark, config, training_manager, overwrite_models)

        self._prediction_data: Dict[Tuple[int, str], Dict[str, pd.DataFrame]] = {}
        self._ground_truth: Dict[Tuple[int, str], Dict[str, pd.DataFrame]] = {}

    @property
    def prediction_data(self) -> Dict[Tuple[int, str], Dict[str, pd.DataFrame]]:
        """Return cached prediction datasets for each step/tag."""

        return self._prediction_data

    @property
    def ground_truth(self) -> Dict[Tuple[int, str], Dict[str, pd.DataFrame]]:
        """Return cached ground truth datasets for each step/tag."""

        return self._ground_truth

    def _load_data(self, step: int, tag: str) -> None:
        """Fetch input data for a given *step* and loader *tag*."""

        start_dt: datetime = self._config.forecast_dates[step][0]
        end_dt: datetime = self._config.forecast_dates[step][-1]

        loaders = self._config.data_loaders
        loader: DataLoader = loaders[tag]

        self._prediction_data[(step, tag)] = loader.get_prediction_data(
            self._spark, start_dt, end_dt
        )
        self._portfolio_data[(step, tag)] = loader.get_portfolio(
            self._spark, self._config.train_ends[step - 1]
        )
        self._ground_truth[(step, tag)] = loader.get_ground_truth(
            self._spark, start_dt, end_dt
        )

    def _create_scenario_data(self, step: int) -> pd.DataFrame:
        dfs: List[pd.DataFrame] = [
            self._prediction_data[(step, tag)]["features"]
            for tag in self._config.data_loaders
        ]

        for df in dfs:
            if "report_dt" not in df.columns:
                df.insert(0, "report_dt", self._config.forecast_dates[step])

        return reduce(
            lambda x, y: x.merge(
                y,
                on="report_dt",
                how="inner",
                suffixes=("", "_copy"),
            ),
            dfs,
        )

    def _create_calculator(self, step: int) -> AbstractCalculator:
        """Instantiate a calculator for the specified *step*."""

        models: Dict[str, ModelInfo] = {
            tag: self.trained_models[(step, tag)] for tag in self._config.trainers
        }

        scenario: Scenario = Scenario(
            portfolio_dt=self._config.train_ends[step - 1],
            horizon=self._config.horizon,
            scenario_data=self._create_scenario_data(step),
        )

        model_data: Dict[str, Any] = {
            f"{tag}_portfolio": self._portfolio_data[(step, tag)]
            for tag in self._config.data_loaders
        }

        calculator_cls = self._config.calculator_type
        if calculator_cls is None:
            raise ValueError("calculator_type must be provided for backtest execution")

        return calculator_cls(
            self.register,
            models,
            scenario,
            model_data,
        )

    def _run_step(self, step: int) -> None:
        """Execute a single backtest step."""

        for tag in self._config.data_loaders:
            self._load_data(step, tag)

        calc = self._create_calculator(step)
        if self._config.calc_type is None:
            raise ValueError("calc_type must be provided for backtest execution")

        self._calc_results[step] = calc.calculate(self._config.calc_type)

    def run_all(self) -> None:
        """Train models and run the backtest for each configured step."""

        self.train_models()
        logger.info("training models completed")

        for step in range(1, self._config.steps + 1):
            self._run_step(step)
            logger.info(f"step {step} completed")

    def _get_backtest_entity(self, df: pd.DataFrame) -> BackTestInfoEntity:
        """Transform chart data into a database persistence entity."""

        if self._config.calculator_type is None:
            raise ValueError("calculator_type must be provided for persistence")

        backtest_info = BackTestInfoEntity(
            first_train_dt=self._config.first_train_end_dt,
            steps=self._config.steps,
            horizon=self._config.horizon,
            calculator=self._config.calculator_type.__name__,
            calc_type=str(self._config.calc_type),
            tag=f"{self._config.calculator_type.__name__}_{self._config.horizon}_{self._config.steps}",
        )

        for record in df.to_dict("records"):
            data = BackTestDataEntity(
                report_dt=record["report_dt"],
                train_last_date=record["train_last_date"],
                pred=record["pred"],
                truth=record["truth"],
                backtest_step=record["backtest_step"],
                periods_ahead=record["periods_ahead"],
                product=record["product"],
                units=record["units"],
                segment=record["segment"],
                tag="",
            )

            backtest_info.backtest_data.append(data)

        return backtest_info

    def save_to_db(
        self,
        model_db: ModelDB,
        analyzers: Iterable[Any],
        segment_map: Optional[Dict[str, str]] = DEPOSIT_SEGMENT_MAP,
    ) -> bool:
        """Persist backtest results in *model_db* using *analyzers*."""

        dfs: List[pd.DataFrame] = []

        for analyzer in analyzers:
            analyzer_instance = analyzer()

            chart_data: Dict[str, pd.DataFrame] = analyzer_instance.get_chart_data(self)

            for _, v in chart_data.items():
                v["units"] = analyzer_instance.units
                dfs.append(v)

        df: pd.DataFrame = pd.concat(dfs)
        mapping: Dict[str, str] = segment_map if segment_map is not None else {}
        df["segment"] = df.apply(
            lambda x: AbstractEngine.get_segment(x["product"], mapping), axis=1
        )
        backtest_info = self._get_backtest_entity(df)
        return model_db.save_backtest(backtest_info)


@dataclass
class BackTestHonestConfig(BaseConfig):
    """Configuration for honest backtests with explicit scenarios."""

    steps: int = 1
    scenario_loader: Optional[DataLoader] = None

    @property
    def train_ends(self) -> List[datetime]:
        """Return the training end dates for each step."""

        return self._train_ends()

    @property
    def forecast_dates(self) -> Dict[int, List[datetime]]:
        """Return the forecast horizons for each step."""

        return self._forecast_dates(self.steps)


class BackTestHonestEngine(AbstractEngine[BackTestHonestConfig]):
    """Engine variant that pulls scenario data from dedicated loaders."""

    def __init__(
        self,
        spark: SparkSession,
        config: BackTestHonestConfig,
        training_manager: TrainingManager,
        overwrite_models: bool = True,
    ) -> None:
        super().__init__(spark, config, training_manager, overwrite_models)

        self._training_manager._overwrite_models = overwrite_models

        self._prediction_data: Dict[Tuple[int, str], Dict[str, pd.DataFrame]] = {}
        self._scenario_data: Dict[int, pd.DataFrame] = {}
        self._portfolio_snapshots: Dict[int, pd.DataFrame] = {}
        self._ground_truth: Dict[int, Dict[str, pd.DataFrame]] = {}

    @property
    def prediction_data(self) -> Dict[Tuple[int, str], Dict[str, pd.DataFrame]]:
        """Return cached prediction data sets."""

        return self._prediction_data

    def _load_ground_truth(self, step: int) -> None:
        """Load ground truth data for *step* from configured loaders."""

        start_dt: datetime = self._config.forecast_dates[step][0]
        end_dt: datetime = self._config.forecast_dates[step][-1]

        self._ground_truth[step] = {
            tag: data_loader.get_ground_truth(self._spark, start_dt, end_dt)
            for tag, data_loader in self._config.data_loaders.items()
        }

    @property
    def ground_truth(self) -> Dict[int, Dict[str, pd.DataFrame]]:
        """Return ground truth data, loading it lazily when required."""

        if not self._ground_truth:
            for step in range(1, self._config.steps + 1):
                self._load_ground_truth(step)

        return self._ground_truth

    def _load_scenario(self, step: int) -> None:
        """Load scenario inputs and portfolio snapshots for *step*."""

        start_dt: datetime = self._config.forecast_dates[step][0]
        end_dt: datetime = self._config.forecast_dates[step][-1]

        portfolio_dt = self._config.train_ends[step - 1]

        loader = self._config.scenario_loader
        if loader is None:
            raise ValueError("scenario_loader must be provided for honest backtest")

        self._portfolio_snapshots[step] = loader.get_portfolio(
            self._spark, portfolio_dt
        )
        self._scenario_data[step] = loader.get_scenario(self._spark, start_dt, end_dt)

    def _create_calc(self, step: int) -> AbstractCalculator:
        """Instantiate the calculator for the honest backtest *step*."""

        dt_: datetime = self._config.train_ends[step - 1]

        models: Dict[str, ModelInfo] = {
            tag: self.trained_models[(step, tag)] for tag in self._config.trainers
        }
        scenario_: Scenario = Scenario(
            dt_,
            self._config.horizon,
            self._scenario_data[step],
        )
        model_data: Dict[str, Any] = {
            "portfolio": {dt_: self._portfolio_snapshots[step]},
            "features": pd.DataFrame().rename_axis(_REPORT_DT_COLUMN),
        }

        calculator_cls = self._config.calculator_type
        if calculator_cls is None:
            raise ValueError("calculator_type must be provided for honest backtest execution")

        return calculator_cls(
            self.register,
            models,
            scenario_,
            model_data,
        )

    def _run_step(self, step: int) -> None:
        """Load inputs and execute a single honest backtest step."""

        self._load_scenario(step)
        self.calc: AbstractCalculator = self._create_calc(step)
        if self._config.calc_type is None:
            raise ValueError("calc_type must be provided for honest backtest execution")

        self._calc_results[step] = self.calc.calculate(self._config.calc_type)

    def run_all(self) -> None:
        """Run the full honest backtest workflow."""

        self.train_models()
        logger.info("training models completed")

        for step in range(1, self._config.steps + 1):
            self._run_step(step)
            logger.info(f"step {step} completed")

    def save_to_db(
        self, model_db: ModelDB, analyzers: List[Any], segment_map: Dict[str, str]
    ) -> None:
        """Persist honest backtest artefacts. Currently a stub."""

        pass
