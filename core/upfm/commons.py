"""Shared domain objects used by the calculator package."""

from __future__ import annotations

from abc import ABC, abstractmethod
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pickle import load
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import pandas as pd  # type: ignore[import-untyped]
from pandas import DataFrame  # type: ignore[import-untyped]
from pandas.tseries.offsets import MonthEnd  # type: ignore[import-untyped]
from dateutil.relativedelta import relativedelta  # type: ignore[import-untyped]
from pyspark.sql import SparkSession  # type: ignore[import-not-found]

_REPORT_DT_COLUMN = "report_dt"


class Products(Enum):
    """Enumeration of supported product identifiers."""

    pass


@dataclass(frozen=True)
class TrainingPeriod:
    """Chronological range describing a model training period."""

    start_dt: Optional[datetime]
    end_dt: Optional[datetime]

    def __str__(self) -> str:
        dateformat = "%Y%m"
        start = self.start_dt.strftime(dateformat) if self.start_dt else ""
        end = f"_{self.end_dt.strftime(dateformat)}" if self.end_dt else ""
        return f"{start}{end}"


@dataclass(frozen=True)
class ModelInfo:
    """Lightweight descriptor uniquely identifying a trained model."""

    prefix: str
    training_period: Optional[TrainingPeriod] = None

    def __str__(self) -> str:
        if self.training_period is not None:
            return f"{self.prefix}_{self.training_period}"
        return f"{self.prefix}_"

    @property
    def model_key(self) -> "ModelInfo":
        """Return a version of the descriptor stripped from training dates."""

        return ModelInfo(self.prefix)

    @staticmethod
    def from_str(
        text: str,
        separator: str = "_",
        dateformat: str = "%Y%m",
    ) -> "ModelInfo":
        """Parse a :class:`ModelInfo` from its serialised representation."""

        parts: List[str] = text.split(separator)
        if len(parts) < 3:
            return ModelInfo(text)

        try:
            from_dt = datetime.strptime(parts[-2], dateformat)
            to_dt = datetime.strptime(parts[-1], dateformat)
        except ValueError:
            return ModelInfo(text)

        training_period = TrainingPeriod(from_dt, to_dt)
        prefix = separator.join(parts[:-2])
        return ModelInfo(prefix, training_period)


class MLException(Exception):
    """Raised when a model reports a domain specific failure."""


class Scenario:
    """Input scenario used to drive model forecasts."""

    def __init__(
        self, portfolio_dt: datetime, horizon: int, scenario_data: DataFrame
    ) -> None:
        self._portfolio_dt: datetime = portfolio_dt
        self._horizon: int = horizon
        self._scenario_data: DataFrame = scenario_data

    @property
    def forecast_dates(self) -> List[datetime]:
        """Return forecast horizon dates derived from the portfolio date."""

        dates_: List[datetime] = [
            self.portfolio_dt + relativedelta(months=m)
            for m in range(1, self._horizon + 1)
        ]
        return [dt.replace(day=monthrange(dt.year, dt.month)[1]) for dt in dates_]

    @property
    def scenario_data(self) -> DataFrame:
        """Return the raw scenario data frame."""

        return self._scenario_data

    @property
    def horizon(self) -> int:
        """Return the forecast horizon expressed in months."""

        return self._horizon

    @property
    def portfolio_dt(self) -> datetime:
        """Return the portfolio snapshot date."""

        return self._portfolio_dt

    def subscenario(self, forecast_date: datetime) -> "Scenario":
        """Create a single-step scenario anchored at ``forecast_date``."""

        return Scenario(
            portfolio_dt=(forecast_date + MonthEnd(-1)).to_pydatetime(),
            horizon=1,
            scenario_data=self._scenario_data.loc[[forecast_date], :],
        )

    def unpivot_data(self, column_prefix: str, col_name: str = "name") -> DataFrame:
        """Return a melted data frame for columns starting with *column_prefix*."""

        column_names = [
            _REPORT_DT_COLUMN,
            *[
                column
                for column in self.scenario_data.columns
                if column.startswith(column_prefix)
            ],
        ]
        return pd.melt(
            self._scenario_data[column_names],
            id_vars=[_REPORT_DT_COLUMN],
            value_name=col_name,
        )


@dataclass
class ForecastContext:
    """Container with contextual information passed to models."""

    portfolio_dt: Optional[datetime] = None
    forecast_horizon: int = 0
    scenario: Optional[Scenario] = None
    model_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def forecast_dates(self) -> List[datetime]:
        if self.scenario is None:
            raise ValueError("Scenario is not set for the forecast context")

        return self.scenario.forecast_dates

    def subcontext(self, forecast_date: datetime) -> "ForecastContext":
        if self.scenario is None:
            raise ValueError("Scenario is not set for the forecast context")

        return ForecastContext(
            portfolio_dt=(forecast_date + MonthEnd(-1)).to_pydatetime(),
            forecast_horizon=1,
            scenario=self.scenario.subscenario(forecast_date),
            model_data=self.model_data,
        )


class BaseModel(ABC):
    """Adapter interface wrapping persistence and prediction logic."""

    def __init__(self, model_info_: ModelInfo, filepath_or_buffer: Any) -> None:
        self._model_info: ModelInfo = model_info_
        self._model_meta: Any = self._unpickle_file_or_buffer(filepath_or_buffer)

    @abstractmethod
    def predict(
        self,
        forecast_context: ForecastContext,
        portfolio: Optional[DataFrame] = None,
        **params: Any,
    ) -> Any:
        """Return a prediction produced by the underlying model."""

    def _unpickle_file_or_buffer(self, filepath_or_buffer: Any) -> Any:
        """Load a pickled model from *filepath_or_buffer*."""

        if hasattr(filepath_or_buffer, "read"):
            return load(filepath_or_buffer)

        with open(filepath_or_buffer, "rb") as in_file:
            return load(in_file)

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info

    @property
    def model(self) -> Any:
        return self._model_meta


class ModelTrainer(ABC):
    """Interface describing the operations required to train a model."""

    @abstractmethod
    def get_trained_model(
        self,
        spark: SparkSession,
        end_date: datetime,
        start_date: Optional[datetime] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Return a trained model instance."""

    @abstractmethod
    def save_trained_model(
        self,
        spark: SparkSession,
        saving_path: str,
        end_date: datetime,
        start_date: Optional[datetime] = None,
        overwrite: bool = True,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Persist a trained model and return its storage identifier."""


class DataLoader(ABC):
    """Abstract base class describing data access operations."""

    @abstractmethod
    def get_maximum_train_range(self, spark: SparkSession) -> Tuple[datetime, datetime]:
        """Return the maximum available training period."""

    @abstractmethod
    def get_training_data(
        self,
        spark: SparkSession,
        start_date: datetime,
        end_date: datetime,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, DataFrame]:
        """Return training datasets required by the model."""

    @abstractmethod
    def get_prediction_data(
        self,
        spark: SparkSession,
        start_date: datetime,
        end_date: datetime,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, DataFrame]:
        """Return feature data required for evaluation or forecasting."""

    @abstractmethod
    def get_ground_truth(
        self,
        spark: SparkSession,
        start_date: datetime,
        end_date: datetime,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, DataFrame]:
        """Return realised target values for the requested period."""

    def get_portfolio(
        self,
        spark: SparkSession,
        report_date: datetime,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, DataFrame]]:
        """Return optional portfolio data used by the calculator."""

        return None

    def get_scenario(
        self,
        spark: SparkSession,
        start_date: datetime,
        end_date: datetime,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, DataFrame]:
        """Return scenario data required for honest backtesting."""

        raise NotImplementedError


@dataclass(frozen=True)
class ModelMetaInfo:
    model_name: str
    model_trainer: ModelTrainer
    data_loader: DataLoader
    adapter: Type[BaseModel]
    segment: Optional[str] = None
    replenishable_flg: Optional[int] = None
    subtraction_flg: Optional[int] = None


class ModelContainer:
    def __init__(
        self,
        models: Optional[Iterable[ModelMetaInfo]] = None,
        model_containers: Optional[Iterable["ModelContainer"]] = None,
    ) -> None:
        self.models: List[ModelMetaInfo] = list(models) if models is not None else []

        if model_containers is not None:
            for model_container in model_containers:
                self.models.extend(model_container.models)

    @property
    def model_names(self) -> List[str]:
        return [model.model_name for model in self.models]

    @property
    def models_dict(self) -> Dict[str, ModelMetaInfo]:
        return {model.model_name: model for model in self.models}

    @property
    def trainers(self) -> Dict[str, ModelTrainer]:
        return {model.model_name: model.model_trainer for model in self.models}

    @property
    def dataloaders(self) -> Dict[str, DataLoader]:
        return {model.model_name: model.data_loader for model in self.models}

    @property
    def adapter_types(self) -> Dict[str, Type[BaseModel]]:
        return {model.model_name: model.adapter for model in self.models}

    def get_model_by_name(self, model_name: str) -> Optional[ModelMetaInfo]:
        return self.models_dict.get(model_name)

    def get_model_by_conditions(self, **conditions: Any) -> Optional[ModelMetaInfo]:
        if not conditions:
            raise KeyError("Conditions are not defined")

        matches = [
            model
            for model in self.models
            if all(getattr(model, cond) == value for cond, value in conditions.items())
        ]

        if len(matches) > 1:
            raise KeyError(f"More than one model has conditions like {conditions}")

        return matches[0] if matches else None


@dataclass(frozen=True)
class PackageMetaInfo:
    models: Tuple[ModelMetaInfo, ...]
    author: str = "upfm"
    email: str = "upfm@vtb.ru"

    def list_models(self):
        print("index \t | \t model_name")
        print("---------------------------------")
        for i, m in enumerate(self.models):
            print(f"{i} \t | \t {m.model_name}")
