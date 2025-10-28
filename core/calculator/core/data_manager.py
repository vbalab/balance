"""Wrappers responsible for sourcing model data for calculators."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from core.calculator.storage.modeldb import ModelDB
from core.upfm.commons import DataLoader


class DataLoaderProxy:  # TODO: check if even used
    """Facade that caches loader results and optionally persists them."""

    def __init__(
        self,
        spark: Any,
        loaders: Dict[str, DataLoader],
        data_db: Optional[ModelDB] = None,
        save_data: bool = True,
    ) -> None:
        self._spark = spark
        self._loaders: Dict[str, DataLoader] = loaders
        self._data_db: Optional[ModelDB] = data_db
        self._save_data: bool = save_data

    def get_prediction_data(self, loader: str, from_dt: datetime, to: datetime) -> Any:
        """Return prediction features for *loader* between *from_dt* and *to*."""

        prediction_data: Any = None
        if self._data_db is not None:
            prediction_data = self._data_db.find_prediction_data(loader, from_dt, to)
        if not prediction_data:
            prediction_data = self._loaders[loader].get_prediction_data(
                self._spark, from_dt, to
            )
            if self._save_data and self._data_db is not None:
                self._data_db.save_prediction_data(loader, from_dt, to, prediction_data)

        return prediction_data

    def get_ground_truth(self, loader: str, from_dt: datetime, to: datetime) -> Any:
        """Return ground truth for *loader* between *from_dt* and *to*."""

        ground_truth: Any = None
        if self._data_db is not None:
            ground_truth = self._data_db.find_ground_truth(loader, from_dt, to)
        if not ground_truth:
            ground_truth = self._loaders[loader].get_ground_truth(
                self._spark, from_dt, to
            )
            if self._save_data and self._data_db is not None:
                self._data_db.save_ground_truth(loader, from_dt, to, ground_truth)

        return ground_truth

    def get_portfolio(self, loader: str, portfolio_dt: datetime) -> Any:
        """Return portfolio snapshot for *loader* at *portfolio_dt*."""

        portfolio: Any = None
        if self._data_db is not None:
            portfolio = self._data_db.find_portfolio(loader, portfolio_dt)
        if not portfolio:
            print(f"portfolio_dt {portfolio_dt}")
            portfolio = self._loaders[loader].get_portfolio(self._spark, portfolio_dt)
            if self._save_data and self._data_db is not None:
                self._data_db.save_portfolio(loader, portfolio_dt, portfolio)

        return portfolio
