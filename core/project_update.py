from core.models import PlanClose
from datetime import datetime
import pandas as pd
import os
from typing import Dict
from core.upfm.commons import ModelTrainer, MLException

from core.calculator.storage import ModelDB
from core.calculator.core import TrainingManager
from core.models import DepositModels

PORTFOLIO_NAME = "portfolio"
MODEL_DB_NAME = "modeldb_test.bin"


class PortfolioException(MLException):
    """Ошибка во время чтения портфеля"""

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass


def save_portfolio(
    spark, portfolio_dt: datetime, portfolio_folder: str, optional_nan_threshold=2e9
):
    port = PlanClose.data_loader.get_portfolio(spark, portfolio_dt)
    balance_nan = port.loc[port["optional_flg"].isna(), "total_generation"].sum()
    # if balance_nan > optional_nan_threshold:
    # raise ValueError(f"""Balance with NaN optional_flg is {balance_nan} it is higher than threshold: {optional_nan_threshold}. Please check the table and fix problems.""")
    port["report_dt"] = pd.to_datetime(port["report_dt"])
    port["optional_flg"] = port.optional_flg.fillna(0)
    file_name = f"{PORTFOLIO_NAME}_{portfolio_dt.strftime('%Y-%m')}.csv"
    full_path = os.path.join(portfolio_folder, file_name)
    port.to_csv(full_path, index=False)
    print(f"Portfolio saved to: {full_path}")


def load_portfolio(portfolio_dt: datetime, folder_path):
    file_name = f"{PORTFOLIO_NAME}_{portfolio_dt.strftime('%Y-%m')}.csv"
    full_path = os.path.join(folder_path, file_name)
    try:
        port = pd.read_csv(full_path, parse_dates=True)
    except FileNotFoundError:
        raise PortfolioException(
            f"Ошибка портфеля! Модель не поддерживает временной период: {portfolio_dt}"
        )

    port["report_dt"] = pd.to_datetime(port["report_dt"])
    return port


def get_model_db(model_db_folder, model_db_name):
    sqlite_filepath = os.path.join(model_db_folder, model_db_name)
    DB_URL = f"sqlite:///{sqlite_filepath}"
    model_db = ModelDB(DB_URL)
    return model_db


def train_models_dt(
    spark,
    train_dt: datetime,
    model_folder: str,
    model_db_name: str = MODEL_DB_NAME,
    trainers_dict: Dict[str, ModelTrainer] = DepositModels.trainers,
):
    print(os.path.abspath(model_folder))
    model_db = get_model_db(model_folder, model_db_name)
    training_manager = TrainingManager(
        spark, trainers=trainers_dict, folder=model_folder, modeldb=model_db
    )
    training_manager.train_models_one_date(train_dt)
