import logging
import pandas as pd
from abc import ABC
from functools import reduce
from datetime import datetime
from dataclasses import dataclass
from pyspark.sql import SparkSession
from typing import Dict, List, Tuple, Any, Type

from core.upfm.commons import (
    DataLoader,
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
from core.calculator.storage import ModelDB, BackTestInfoEntity, BackTestDataEntity


logging.config.dictConfig(Settings.LOGGING_CONFIG)
logger = logging.getLogger("core")


@dataclass
class BackTestConfig(BaseConfig):
    steps: int = 1

    @property
    def train_ends(self) -> List[datetime]:
        return self._train_ends()

    @property
    def forecast_dates(self) -> Dict[int, List[datetime]]:
        return self._forecast_dates(self.steps)


class BackTestEngine(AbstractEngine):
    # TODO: better move to definitions.py
    DEPOSIT_SEGMENT_MAP = {
        "_portfolio": "portfolio",
        "_novip_": "novip",
        "_vip_": "vip",
    }

    def __init__(
        self,
        spark: SparkSession,
        config: BackTestConfig,
        training_manager: TrainingManager,
        overwrite_models=True,
    ) -> None:
        super().__init__(spark, config, training_manager, overwrite_models)

        self._prediction_data: Dict[Tuple[int, str], Dict[str, pd.DataFrame]] = {}
        self._ground_truth: Dict[Tuple[int, str], Dict[str, pd.DataFrame]] = {}

    @property
    def prediction_data(self):
        return self._prediction_data

    @property
    def ground_truth(self):
        return self._ground_truth

    def _load_data(self, step: int, tag: str) -> None:
        start_dt: datetime = self._config.forecast_dates[step][0]
        end_dt: datetime = self._config.forecast_dates[step][-1]

        loader: DataLoader = self._config.data_loaders[tag]

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
        models = {
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

        return self._config.calculator_type(
            self.register,
            models,
            scenario,
            model_data,
        )

    def _run_step(self, step: int):
        for tag in self._config.data_loaders:
            self._load_data(step, tag)

        calc = self._create_calculator(step)
        self._calc_results[step] = calc.calculate(self._config.calc_type)

    def run_all(self) -> None:
        self.train_models()
        logger.info("training models completed")

        for step in range(1, self._config.steps + 1):
            self._run_step(step)
            logger.info(f"step {step} completed")

    def _get_backtest_entity(self, df: pd.DataFrame) -> BackTestInfoEntity:
        backtest_info = BackTestInfoEntity(
            first_train_dt=self._config.first_train_end_dt,
            steps=self._config.steps,
            horizon=self._config.horizon,
            calculator=self._config.calculator_type.__name__,
            calc_type=str(self._config.calc_type),
            tag=f"{b_info.calculator}_{b_info.horizon}_{b_info.steps}",
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
        analyzers: List[Any],
        segment_map: Dict[str, str] = DEPOSIT_SEGMENT_MAP,
    ) -> bool:
        dfs: List[pd.DataFrame] = []

        for analyzer in analyzers:
            analyzer_instance = analyzer()

            chart_data: Dict[str, pd.DataFrame] = analyzer_instance.get_chart_data(self)

            for _, v in chart_data.items():
                v["units"] = analyzer_instance.units
                dfs.append(v)

        df: pd.DataFrame = pd.concat(dfs)
        df["segment"] = df.apply(
            lambda x: AbstractEngine.get_segment(x["product"], segment_map), axis=1
        )
        backtest_info = self._get_backtest_entity(df)
        return model_db.save_backtest(backtest_info)


@dataclass
class BackTestHonestConfig(BaseConfig):
    steps: int = 1
    scenario_loader: DataLoader = None

    @property
    def train_ends(self) -> List[datetime]:
        return self._train_ends()

    @property
    def forecast_dates(self) -> Dict[int, List[datetime]]:
        return self._forecast_dates(self.steps)


class BackTestHonestEngine(AbstractEngine):
    def __init__(
        self,
        spark: SparkSession,
        config: BackTestConfig,
        training_manager: TrainingManager,
        overwrite_models=True,
    ) -> None:
        super().__init__(spark, config, training_manager, overwrite_models)

        self._training_manager._overwrite_models = overwrite_models

        self._scenario_data: Dict[int, pd.DataFrame] = {}
        self._portfolio_data: Dict[int, pd.DataFrame] = {}
        self._ground_truth: Dict[Tuple[int, str], Dict[str, pd.DataFrame]] = {}

    @property
    def prediction_data(self):
        return self._prediction_data

    def _load_ground_truth(self, step):
        start_dt: datetime = self._config.forecast_dates[step][0]
        end_dt: datetime = self._config.forecast_dates[step][-1]

        self._ground_truth[step] = {
            tag: data_loader.get_ground_truth(self._spark, start_dt, end_dt)
            for tag, data_loader in self._config.data_loaders.items()
        }

    @property
    def ground_truth(self):
        if not self._ground_truth:
            for step in range(1, self._config.steps + 1):
                self._load_ground_truth(step)

        return self._ground_truth

    def _load_scenario(self, step):
        start_dt: datetime = self._config.forecast_dates[step][0]
        end_dt: datetime = self._config.forecast_dates[step][-1]

        portfolio_dt = self._config.train_ends[step - 1]

        loader: DataLoader = self._config.scenario_loader

        self._portfolio_data[step] = loader.get_portfolio(self._spark, portfolio_dt)
        self._scenario_data[step] = loader.get_scenario(self._spark, start_dt, end_dt)

    def _create_calc(self, step: int) -> AbstractCalculator:
        dt_: datetime = self._config.train_ends[step - 1]

        models = {
            tag: self.trained_models[(step, tag)] for tag in self._config.trainers
        }
        scenario_: Scenario = Scenario(
            dt_,
            self._config.horizon,
            self._scenario_data[step],
        )
        model_data: Dict[str, Any] = {
            "portfolio": {dt_: self._portfolio_data[step]},
            "features": pd.DataFrame().rename_axis(_REPORT_DT_COLUMN),
        }

        return self._config.calculator_type(
            self.register,
            models,
            scenario_,
            model_data,
        )

    def _run_step(self, step: int):
        self._load_scenario(step)
        self.calc: AbstractCalculator = self._create_calc(step)
        self._calc_results[step] = self.calc.calculate(self._config.calc_type)

    def run_all(self) -> None:
        self.train_models()
        logger.info("training models completed")

        for step in range(1, self._config.steps + 1):
            self._run_step(step)
            logger.info(f"step {step} completed")

    def save_to_db(
        self, model_db: ModelDB, analyzers: List[Any], segment_map: Dict[str, str]
    ) -> None:
        pass
