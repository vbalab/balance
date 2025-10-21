import logging
from os import path
from glob import glob
from io import BytesIO
from os import makedirs
from datetime import datetime
from pyspark.sql import SparkSession
from typing import Dict, Optional, List, Tuple

from core.calculator.storage import ModelDB
from core.upfm.commons import ModelTrainer, ModelInfo
from core.calculator.core import Settings, ModelRegister

logging.config.dictConfig(Settings.LOGGING_CONFIG)
logger = logging.getLogger("core")


class TrainingManager:
    def __init__(
        self,
        spark: SparkSession,
        trainers: Dict[str, ModelTrainer],
        folder: str,
        modeldb: Optional[ModelDB] = None,
        force_training: bool = False,
        overwrite_models: bool = True,
    ) -> None:
        self._spark = spark
        self._trainers = trainers
        self._db: ModelDB = modeldb
        self._model_folder = folder
        self._force_training: bool = force_training
        self._overwrite_models = overwrite_models

        self._end_dates: Dict[int, datetime] = {}
        self._trained_models: Dict[Tuple[int, str], ModelInfo] = {}
        self._model_data: Dict[ModelInfo, BytesIO] = {}

    def _find_missing_models(self) -> List[Tuple[str, int]]:
        if not self._db:
            return []

        missing_in_db: List[Tuple[str, int]] = []

        for tag in self._trainers:
            for step in self._end_dates:
                if not self._db.find_trained_model_by_dt(tag, self._end_dates[step]):
                    missing_in_db.append((tag, step))

        return missing_in_db

    def _train_model(self, tag: str, step: int) -> bool:
        logger.info(
            f"train_model tag={tag}, step={step}, end_date={self._end_dates[step]}"
        )

        try:
            trainer: ModelTrainer = self._trainers[tag]
            end_dt: datetime = self._end_dates[step]
            model_path: str = trainer.save_trained_model(
                self._spark,
                self._model_folder,
                end_dt,
                overwrite=self._overwrite_models,
            )

            if model_path:
                self._trained_models[(step, tag)] = ModelInfo.from_str(
                    path.splitext(model_path)[0]
                )

                if self._db:
                    self._db.save_trained_model(
                        path.join(self._model_folder, model_path),
                        True,
                    )
        except Exception as e:
            logger.exception(f"failed to train the model {tag}")
            return False

        return True

    def _train_missing_models(self) -> None:
        missing_in_db = self._find_missing_models()
        logger.info(f"missing models: {missing_in_db}")

        for missing in missing_in_db:
            logger.info(f"train {missing[0]} {missing[1]}")
            self._train_model(missing[0], missing[1])

    def _train_models(self) -> None:
        for tag in self._trainers:
            for step in self._end_dates:
                self._train_model(tag, step)

    def _create_folders(self) -> None:
        makedirs(self._model_folder, exist_ok=True)

    def _load_models(self):
        if not self._db:
            raise ValueError(f"Model db is None, use force_training")

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
        self._create_folders()

        self._end_dates: Dict[int, datetime] = dict(enumerate(end_dates, start=1))

        if self._force_training:
            self._train_models()
            register.add_models_from_folder(self._model_folder)
        else:
            self._train_missing_models()
            self._load_models()
            register.add_models_from_bytes(self._model_data)

    def get_models_by_step(self, step: int):
        return {
            tag: self._trained_models[(step, tag)].to_model_info()
            for tag in self._trainers
        }

    @property
    def trained_models(self) -> Dict[Tuple[int, str], ModelInfo]:
        return self._trained_models

    def train_models_one_date(self, end_dt: datetime):
        self._create_folders()
        self._end_dates: Dict[int, datetime] = {1: end_dt}

        if self._force_training:
            self._train_models()
        else:
            self._train_missing_models()
