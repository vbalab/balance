from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import pandas as pd
from pyspark.sql import SparkSession

from core.calculator.storage import ModelDB
from core.upfm.commons import BaseModel, DataLoader, ModelInfo, ModelTrainer
from core.calculator.core import (
    AbstractCalculator,
    CalculationResult,
    CalculationType,
    ModelRegister,
    Settings,
    TrainingManager,
)

logging.config.dictConfig(Settings.LOGGING_CONFIG)
logger = logging.getLogger("core")

# TODO: Create Engine for 1 model with prefix, to not make that many `Dict`s, make BaseConfig Dict[str, Engine]


@dataclass
class BaseConfig:
    first_train_end_dt: Optional[datetime] = None
    horizon: int = 1  # TODO: check if necessary here
    trainers: Optional[Dict[str, ModelTrainer]] = None
    model_params: Optional[Dict[str, Dict[str, ModelTrainer]]] = None
    data_loaders: Optional[Dict[str, DataLoader]] = None
    calculator_type: Optional[Type[AbstractCalculator]] = None
    calc_type: Optional[CalculationType] = None
    adapter_types: Optional[Dict[str, Type[BaseModel]]] = None

    @property
    def first_dt_str(self) -> str:
        return self.first_train_end_dt.strftime("%Y-%m-%d")

    def _get_dates(self) -> List[datetime]:
        datarange = pd.date_range(
            self.first_train_end_dt,
            periods=self.horizon * steps_ + 1,
            freq="M",
        )

        dates: List[datetime] = list(map(lambda x: x.to_pydatetime(), datarange))
        return dates

    def _train_ends(self) -> List[datetime]:
        dates = _get_dates()

        return dates[:: self.horizon][:-1]

    def _forecast_dates(self, steps_: int = 1) -> Dict[int, List[datetime]]:
        dates = _get_dates()

        return {
            s: dates[1 + (s - 1) * self.horizon : 1 + s * self.horizon]
            for s in range(1, steps_ + 1)
        }


T = TypeVar("T", bound="AbstractEngine")


class AbstractEngine(ABC):
    """
    Класс позволяет в ходе его работы обучать нужные для бэктеста модели и
    запускать калькулятор с переданными параметрами, получать итоговый результат.
    """

    def __init__(
        self,
        spark: SparkSession,
        # TODO: double work:
        config: BaseConfig,  # BaseConfig -> trainers: Dict[str, ModelTrainer]
        training_manager: TrainingManager,  # TrainingManager -> trainers: Dict[str, ModelTrainer]
        overwrite_models=True,
    ) -> None:
        self._spark: SparkSession = spark
        self._config: BaseConfig = config
        self._overwrite_models: bool = overwrite_models
        self._training_manager: TrainingManager = training_manager

        self._register: ModelRegister = ModelRegister(
            adapter_types=self._config.adapter_types
        )
        self._portfolio_data: Dict[Tuple[int, str], pd.DataFrame] = {}
        self._calc_results: Dict[int, CalculationResult] = {}

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()

        to_exclude = {"_spark", "_training_manager"}
        state = {k: state[k] for k in set(state.keys()) - to_exclude}

        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    def save(self, path: str) -> None:
        with open(path, "wb") as fo:
            pickle.dump(self, fo)

    #     def _get_backtest_entity(self, df: pd.DataFrame) -> BackTestInfoEntity:
    #         b_info = BackTestInfoEntity()
    #         b_info.first_train_dt = self._config.first_train_end_dt
    #         b_info.steps = self._config.steps
    #         b_info.horizon = self._config.horizon
    #         b_info.calculator = self._config.calculator_type.__name__
    #         b_info.calc_type = str(self._config.calc_type)
    #         b_info.tag = f'{b_info.calculator}_{b_info.horizon}_{b_info.steps}'
    #         for r in df.to_dict('records'):
    #             d: BackTestDataEntity = BackTestDataEntity()
    #             d.report_dt = r['report_dt']
    #             d.train_last_date = r['train_last_date']
    #             d.pred = r['pred']
    #             d.truth = r['truth']
    #             d.backtest_step = r['backtest_step']
    #             d.periods_ahead = r['periods_ahead']
    #             d.product = r['product']
    #             d.units = r['units']
    #             d.segment = r['segment']
    #             d.tag = ''
    #             b_info.backtest_data.append(d)

    #         return b_info

    #     def save_to_db(self, model_db: ModelDB, analyzers: List[Any], segment_map: Dict[str, str] = DEPOSIT_SEGMENT_MAP) -> bool:
    #         dfs: List[pd.DataFrame] = []
    #         for an in analyzers:
    #             a_inst: Any = an()
    #             chart_data: Dict[str, pd.DataFrame] = a_inst.get_chart_data(self)
    #             for _, v in chart_data.items():
    #                 v['units'] = a_inst.units
    #                 dfs.append(v)

    #         df: pd.DataFrame = pd.concat(dfs)
    #         df['segment'] = df.apply(lambda x: AbstractEngine.get_segment(x['product'], segment_map), axis=1)
    #         backtest_info = self._get_backtest_entity(df)
    #         return model_db.save_backtest(backtest_info)

    @staticmethod
    def get_segment(product: str, mapping: Dict[str, str]) -> str:
        prod_l: str = product.lower()

        for k, v in mapping.items():
            if prod_l.find(k) > 0:
                return v

        return ""

    @classmethod
    def load_from_file(cls, path: str) -> T:
        with open(path, "rb") as fo:
            return pickle.load(fo)

    @property
    def register(self) -> ModelRegister:
        return self._register

    @property
    def trained_models(self) -> Dict[Tuple[int, str], ModelInfo]:
        return self._training_manager.trained_models

    @property
    def portfolio_data(self) -> Dict[Tuple[int, str], pd.DataFrame]:
        return self._portfolio_data

    @property
    def calc_results(self) -> Dict[int, CalculationResult]:
        return self._calc_results

    def train_models(self) -> None:
        self._training_manager.add_to_register(self._register, self._config.train_ends)

    @abstractmethod
    def run_all(self) -> None:
        pass

    @abstractmethod
    def save_to_db(
        self, model_db: ModelDB, analyzers: List[Any], segment_map: Dict[str, str]
    ) -> Any:
        pass
