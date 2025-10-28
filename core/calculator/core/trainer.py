"""Utilities for training and registering forecasting models."""

from __future__ import annotations

import logging
import logging.config
from datetime import datetime
from io import BytesIO
from os import makedirs, path
from typing import Dict, List, Optional, Tuple

from pyspark.sql import SparkSession  # type: ignore[import-not-found]

from core.calculator.storage import ModelDB
from core.upfm.commons import ModelInfo, ModelTrainer
from core.calculator.core import Settings, ModelRegister

logging.config.dictConfig(Settings.LOGGING_CONFIG)
logger = logging.getLogger("core")


class TrainingManager:
    """Coordinate trainer instances, storage, and model registration."""

    def __init__(
        self,
        spark: SparkSession,
        trainers: Dict[str, ModelTrainer],
        folder: str,
        modeldb: Optional[ModelDB] = None,
        force_training: bool = False,
        overwrite_models: bool = True,
    ) -> None:
        self._spark: SparkSession = spark
        self._trainers: Dict[str, ModelTrainer] = trainers
        self._db: Optional[ModelDB] = modeldb
        self._model_folder: str = folder
        self._force_training: bool = force_training
        self._overwrite_models: bool = overwrite_models

        self._end_dates: Dict[int, datetime] = {}
        self._trained_models: Dict[Tuple[int, str], ModelInfo] = {}
        self._model_data: Dict[ModelInfo, BytesIO] = {}

    def _find_missing_models(self) -> List[Tuple[str, int]]:
        """Return (tag, step) tuples missing from the model database."""

        if not self._db:
            return []

        missing_in_db: List[Tuple[str, int]] = []

        for tag in self._trainers:
            for step in self._end_dates:
                if not self._db.find_trained_model_by_dt(tag, self._end_dates[step]):
                    missing_in_db.append((tag, step))

        return missing_in_db

    def _train_model(self, tag: str, step: int) -> bool:
        """Train and persist the model identified by *tag* and *step*."""

        logger.info(
            f"train_model tag={tag}, step={step}, end_date={self._end_dates[step]}"
        )

        try:
            trainer: ModelTrainer = self._trainers[tag]
            end_dt: datetime = self._end_dates[step]
            model_path = trainer.save_trained_model(
                self._spark,
                self._model_folder,
                end_dt,
                overwrite=self._overwrite_models,
            )

            if model_path:
                model_info = ModelInfo.from_str(path.splitext(model_path)[0])
                if model_info.training_period is None:
                    raise ValueError(
                        f"Model path {model_path} does not encode a training period"
                    )

                self._trained_models[(step, tag)] = model_info

                if self._db:
                    self._db.save_trained_model(
                        path.join(self._model_folder, model_path),
                        True,
                    )
        except Exception:
            logger.exception(f"failed to train the model {tag}")
            return False

        return True

    def _train_missing_models(self) -> None:
        """Train models that are not yet stored in the database."""

        missing_in_db = self._find_missing_models()
        logger.info(f"missing models: {missing_in_db}")

        for missing in missing_in_db:
            logger.info(f"train {missing[0]} {missing[1]}")
            self._train_model(missing[0], missing[1])

    def _train_models(self) -> None:
        """Train every configured model for all end dates."""

        for tag in self._trainers:
            for step in self._end_dates:
                self._train_model(tag, step)

    def _create_folders(self) -> None:
        """Ensure the model output directory exists."""

        makedirs(self._model_folder, exist_ok=True)

    def _load_models(self) -> None:
        """Load existing models from the database into memory."""

        if not self._db:
            raise ValueError("Model db is None, use force_training")

        for tag in self._trainers:
            for step in self._end_dates:
                entity = self._db.find_trained_model_by_dt(tag, self._end_dates[step])
                if entity:
                    self._trained_models[(step, tag)] = entity.to_model_info()

                    # TODO: разобраться с model_data внутри entity
                    self._model_data[entity.to_model_info()] = BytesIO(
                        entity.model_data
                    )

    def add_to_register(
        self,
        register: ModelRegister,
        end_dates: List[datetime],
    ) -> None:
        """Populate *register* with models trained for *end_dates*."""

        self._create_folders()

        self._end_dates = dict(enumerate(end_dates, start=1))

        if self._force_training:
            self._train_models()
            register.add_models_from_folder(self._model_folder)
        else:
            self._train_missing_models()
            self._load_models()
            register.add_models_from_bytes(self._model_data)

    def get_models_by_step(self, step: int) -> Dict[str, ModelInfo]:
        """Return the models trained for a given *step*."""

        return {tag: self._trained_models[(step, tag)] for tag in self._trainers}

    @property
    def trained_models(self) -> Dict[Tuple[int, str], ModelInfo]:
        """Expose the mapping of (step, tag) to :class:`ModelInfo`."""

        return self._trained_models

    def train_models_one_date(self, end_dt: datetime) -> None:
        """Train models for a single *end_dt* cutoff."""

        self._create_folders()
        self._end_dates = {1: end_dt}

        if self._force_training:
            self._train_models()
        else:
            self._train_missing_models()
