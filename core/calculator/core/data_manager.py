from datetime import datetime
from typing import Dict, Optional, Any

from core.calculator.storage.modeldb import ModelDB
from core.upfm.commons import DataLoader


class DataLoaderProxy:  # TODO: check if even used
    def __init__(
        self,
        spark: Any,
        loaders: Dict[str, DataLoader],
        data_db: Optional[ModelDB] = None,
        save_data: bool = True,
    ) -> None:
        self._spark = spark
        self._loaders: Dict[str, DataLaoder] = loaders
        self._data_db: ModelDB = data_db
        self._save_data = save_data

    def get_prediction_data(self, loader: str, from_dt: datetime, to: datetime):
        prediction_data = None
        if self._data_db:
            prediction_data = data_db.find_prediction_data(loader, from_dt, to)
        if not prediction_data:
            prediction_data = self._loaders[loader].get_prediction_data(
                self._spark, from_dt, to
            )
            if self._save_data and self._data_db:
                data_db.save_prediction_data(loader, from_dt, to, prediction_data)

        return prediction_data

    def get_ground_truth(self, loader: str, from_dt: datetime, to: datetime):
        ground_truth = None
        if self._data_db:
            ground_truth = data_db.find_ground_truth(loader, from_dt, to)
        if not ground_truth:
            ground_truth = self._loaders[loader].get_ground_truth(
                self._spark, from_dt, to
            )
            if self._save_data and self._data_db:
                data_db.save_ground_truth(loader, from_dt, to, prediction_data)

        return ground_truth

    def get_portfolio(self, loader: str, portfolio_dt: datetime):
        portfolio = None
        if self._data_db:
            portfolio = data_db.find_portfolio(loader, portfolio_dt)
        if not portfolio:
            print(f"portfolio_dt {portfolio_dt}")
            portfolio = self._loaders[loader].get_portfolio(self._spark, portfolio_dt)
            if self._save_data and self._data_db:
                data_db.save_portfolio(loader, portfolio_dt, portfolio)

        return portfolio
