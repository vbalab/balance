import sys
sys.path.append('..')
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
from datetime import datetime
import os
import importlib

pd.set_option('display.max.columns', 300)

from core.calculator.storage import ModelDB
from core.calculator.core import ForecastConfig
from core.calculator.core import TrainingManager
from core.calculator.core import ForecastConfig
from core.calculator.core import ForecastEngine

from core.calculator.deposits import DepositsCalculationType
from core.calculator.deposits import DepositIterativeCalculator

from core.definitions import *
from core.project_update import load_portfolio

from core.models import DepositModels
from warnings import filterwarnings
filterwarnings('ignore')



train_end = datetime(year=2022, month=12, day=31)


horizon = 3



scenario_data = {

     'expected_amount':      [np.nan for h in range(horizon)],
 
     'SSV':                  [0.48 for h in range(horizon)],
 
     'FOR':                  [4.5 for h in range(horizon)],
    
     'VTB_ftp_rate_[90d]':   [12.3 for h in range(horizon)],
     'VTB_ftp_rate_[180d]':  [12 for h in range(horizon)],
     'VTB_ftp_rate_[365d]':  [12 for h in range(horizon)],
     'VTB_ftp_rate_[548d]':  [12 for h in range(horizon)],
     'VTB_ftp_rate_[730d]':  [12 for h in range(horizon)],
     'VTB_ftp_rate_[1095d]': [12 for h in range(horizon)],
    

     'margin_[90d]':         [0.1 for h in range(horizon)],
     'margin_[180d]':        [0.1 for h in range(horizon)],
     'margin_[365d]':        [0.1 for h in range(horizon)],
     'margin_[548d]':        [0.1 for h in range(horizon)],
     'margin_[730d]':        [0.2 for h in range(horizon)],
     'margin_[1095d]':       [0.2 for h in range(horizon)],
    

     'priv_spread':          [0.4 for h in range(horizon)],

     'vip_spread':           [0.8 for h in range(horizon)],
    

     'r0s1_spread':          [-1 for h in range(horizon)],
    

     'r1s0_spread':          [-1 for h in range(horizon)],
    

     'r1s1_spread':          [-1.2 for h in range(horizon)],
    

     'SBER_max_rate':        [11.2, 11.2, 11.2],
    

     'SA_rate':              [5 for h in range(horizon)]
}
scenario_df_user = pd.DataFrame(scenario_data)

scenario_df = preprocess_scenario(scenario_df_user, train_end, horizon)

port_folder = '../data/portfolio_data'
portfolio = load_portfolio(train_end, port_folder)


from core.models.utils import run_spark_session


spark = None 


folder = '../data/trained_models'

sqlite_filepath = os.path.join(folder, 'modeldb_test.bin')

DB_URL = f"sqlite:///{sqlite_filepath}"
model_db = ModelDB(DB_URL)


config: ForecastConfig = ForecastConfig(
    first_train_end_dt = train_end,
    horizon = horizon,
    trainers = DepositModels.trainers,
    data_loaders = DepositModels.dataloaders,
    calculator_type = DepositIterativeCalculator,
    calc_type = DepositsCalculationType,
    adapter_types = DepositModels.adapter_types,
    scenario_data = scenario_df,
    portfolio = portfolio
)
    
training_manager = TrainingManager(spark, config.trainers, folder, model_db)   
engine: ForecastEngine = ForecastEngine(spark, config, training_manager, overwrite_models=False)
    
    
engine.run_all()