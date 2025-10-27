"""Deposit forecast calculators that orchestrate multiple component models."""

from __future__ import annotations

from enum import auto
from typing import Dict, Optional

import pandas as pd

from upfm.commons import ModelInfo, BaseModel
from deposit_outflow_plan_rur.meta import package_meta_info as meta_outflow
from competitors_reaction.meta import package_meta_info as meta_competitors
from deposit_maturity_structure_rur.meta import package_meta_info as meta_maturity
from deposit_size_structure_rur.meta import package_meta_info as meta_size
from deposit_newbusiness_rur.meta import package_meta_info as meta_newbusiness
from deposit_early_redemption.meta import package_meta_info as meta_redemption
from deposit_renewal_rur.meta import package_meta_info as meta_renewal

from core.calculator.core.calc_base import (
    ModelRegister,
    CalculationType,
    AbstractCalculator,
    CalculationResult,
)


class RetailDepositsCalculationType(CalculationType):
    """Calculation type enumerations for retail deposit forecasts."""

    RetailDeposits = auto()


class DepositNewbusinessCalculator(AbstractCalculator):
    """Run the retail deposit suite of models and assemble their outputs."""

    def __init__(
        self,
        model_register: ModelRegister,
        models: Dict[str, ModelInfo],
        scenario: pd.DataFrame,
        model_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        """Initialise calculator with model registry and scenario context."""

        super().__init__(model_register, models, scenario, model_data)
        self.outflow_plan_model: BaseModel = model_register.get_model(
            self._models[meta_outflow.models[0].model_name]
        )

        self.competitors_model: BaseModel = model_register.get_model(
            self._models[meta_competitors.models[0].model_name]
        )

        self.maturity_structure_novip_noopt_model: BaseModel = model_register.get_model(
            self._models[meta_maturity.models[0].model_name]
        )
        self.maturity_structure_novip_opt_model: BaseModel = model_register.get_model(
            self._models[meta_maturity.models[1].model_name]
        )
        self.maturity_structure_vip_noopt_model: BaseModel = model_register.get_model(
            self._models[meta_maturity.models[2].model_name]
        )
        self.maturity_structure_vip_opt_model: BaseModel = model_register.get_model(
            self._models[meta_maturity.models[3].model_name]
        )

        self.size_structure_novip_noopt_model: BaseModel = model_register.get_model(
            self._models[meta_size.models[0].model_name]
        )
        self.size_structure_novip_opt_model: BaseModel = model_register.get_model(
            self._models[meta_size.models[1].model_name]
        )
        self.size_structure_vip_noopt_model: BaseModel = model_register.get_model(
            self._models[meta_size.models[2].model_name]
        )
        self.size_structure_vip_opt_model: BaseModel = model_register.get_model(
            self._models[meta_size.models[3].model_name]
        )

        self.newbusiness_novip_noopt_model: BaseModel = model_register.get_model(
            self._models[meta_newbusiness.models[0].model_name]
        )
        self.newbusiness_novip_opt_model: BaseModel = model_register.get_model(
            self._models[meta_newbusiness.models[1].model_name]
        )
        self.newbusiness_vip_noopt_model: BaseModel = model_register.get_model(
            self._models[meta_newbusiness.models[2].model_name]
        )
        self.newbusiness_vip_opt_model: BaseModel = model_register.get_model(
            self._models[meta_newbusiness.models[3].model_name]
        )

        self.early_redemption_noopt_model: BaseModel = model_register.get_model(
            self._models[meta_redemption.models[0].model_name]
        )
        self.early_redemption_opt_model: BaseModel = model_register.get_model(
            self._models[meta_redemption.models[1].model_name]
        )

        self.renewal_model: BaseModel = model_register.get_model(
            self._models[meta_renewal.models[0].model_name]
        )

    def calculate_plan_outflows(self) -> pd.DataFrame:
        """Predict plan outflows and persist them in the forecast context."""

        outflows: pd.DataFrame = self.outflow_plan_model.predict(self._forecast_context)
        self._forecast_context.model_data["OUTFLOW_PLAN"] = outflows
        return outflows

    def calculate_competitors(self) -> pd.DataFrame:
        """Predict competitor rates and store them in the context."""

        competitor_rates: pd.DataFrame = self.competitors_model.predict(
            self._forecast_context
        )
        self._forecast_context.model_data["COMPETITOR_RATES"] = competitor_rates
        return competitor_rates

    def calculate_maturity_structure(self) -> Dict[str, pd.DataFrame]:
        """Predict maturity structures for every segment and option flag."""

        maturity_structure_model_key = "MATURITY_STRUCTURE"

        maturity_structure_novip_noopt: pd.DataFrame = (
            self.maturity_structure_novip_noopt_model.predict(self._forecast_context)
        )
        maturity_structure_novip_opt: pd.DataFrame = (
            self.maturity_structure_novip_opt_model.predict(self._forecast_context)
        )
        maturity_structure_vip_noopt: pd.DataFrame = (
            self.maturity_structure_vip_noopt_model.predict(self._forecast_context)
        )
        maturity_structure_vip_opt: pd.DataFrame = (
            self.maturity_structure_vip_opt_model.predict(self._forecast_context)
        )

        maturitystructure: Dict[str, pd.DataFrame] = {
            "NOVIP_NOOPT": maturity_structure_novip_noopt,
            "NOVIP_OPT": maturity_structure_novip_opt,
            "VIP_NOOPT": maturity_structure_vip_noopt,
            "VIP_OPT": maturity_structure_vip_opt,
        }

        self._forecast_context.model_data[maturity_structure_model_key] = (
            maturitystructure
        )
        return maturitystructure

    def calculate_size_structure(self) -> Dict[str, pd.DataFrame]:
        """Predict size structures for every segment and option flag."""

        size_structure_model_key = "SIZE_STRUCTURE"

        size_structure_novip_noopt: pd.DataFrame = (
            self.size_structure_novip_noopt_model.predict(self._forecast_context)
        )
        size_structure_novip_opt: pd.DataFrame = (
            self.size_structure_novip_opt_model.predict(self._forecast_context)
        )
        size_structure_vip_noopt: pd.DataFrame = (
            self.size_structure_vip_noopt_model.predict(self._forecast_context)
        )
        size_structure_vip_opt: pd.DataFrame = (
            self.size_structure_vip_opt_model.predict(self._forecast_context)
        )

        sizestructure: Dict[str, pd.DataFrame] = {
            "NOVIP_NOOPT": size_structure_novip_noopt,
            "NOVIP_OPT": size_structure_novip_opt,
            "VIP_NOOPT": size_structure_vip_noopt,
            "VIP_OPT": size_structure_vip_opt,
        }

        self._forecast_context.model_data[size_structure_model_key] = sizestructure
        return sizestructure

    def calculate_deposit_newbusiness(self) -> Dict[str, pd.DataFrame]:
        """Predict new business volumes for all combinations of segments."""

        newbusiness_model_key = "NEWBUSINESS"
        newbusiness_novip_noopt: pd.DataFrame = (
            self.newbusiness_novip_noopt_model.predict(self._forecast_context)
        )
        newbusiness_novip_opt: pd.DataFrame = self.newbusiness_novip_opt_model.predict(
            self._forecast_context
        )
        newbusiness_vip_noopt: pd.DataFrame = self.newbusiness_vip_noopt_model.predict(
            self._forecast_context
        )
        newbusiness_vip_opt: pd.DataFrame = self.newbusiness_vip_opt_model.predict(
            self._forecast_context
        )
        newbusiness = {
            "NOVIP_NOOPT": newbusiness_novip_noopt,
            "NOVIP_OPT": newbusiness_novip_opt,
            "VIP_NOOPT": newbusiness_vip_noopt,
            "VIP_OPT": newbusiness_vip_opt,
        }
        self._forecast_context.model_data[newbusiness_model_key] = newbusiness
        return newbusiness

    def calculate_early_redemption(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Predict early redemption flows and aggregate by portfolio slices."""

        early_redemption_model_key = "EARLY_REDEMPTION"
        early_redemption_portfolio_key = "ER_PORTFOLIO"
        early_redemption_noopt_res: Dict[str, pd.DataFrame] = (
            self.early_redemption_noopt_model.predict(self._forecast_context)
        )
        cols_noopt = list(
            set(early_redemption_noopt_res["current_portfolio"].columns)
            & set(early_redemption_noopt_res["newbiz_portfolio"].columns)
        )
        early_redemption_noopt_portfolio = pd.concat(
            [
                early_redemption_noopt_res["current_portfolio"][cols_noopt],
                early_redemption_noopt_res["newbiz_portfolio"][cols_noopt],
            ],
            axis=0,
        ).reset_index(drop=True)

        early_redemption_opt_res: Dict[str, pd.DataFrame] = (
            self.early_redemption_opt_model.predict(self._forecast_context)
        )
        cols_opt = list(
            set(early_redemption_opt_res["current_portfolio"].columns)
            & set(early_redemption_opt_res["newbiz_portfolio"].columns)
        )
        early_redemption_opt_portfolio = pd.concat(
            [
                early_redemption_opt_res["current_portfolio"][cols_opt],
                early_redemption_opt_res["newbiz_portfolio"][cols_opt],
            ],
            axis=0,
        ).reset_index(drop=True)

        early_redemption_portfolio = pd.concat(
            [early_redemption_noopt_portfolio, early_redemption_opt_portfolio], axis=0
        ).reset_index(drop=True)
        early_redemption = {
            "NOOPT": early_redemption_noopt_res,
            "OPT": early_redemption_opt_res,
        }
        self._forecast_context.model_data[early_redemption_model_key] = early_redemption
        self._forecast_context.model_data[early_redemption_portfolio_key] = (
            early_redemption_portfolio
        )
        return early_redemption

    def calculate_renewal(self) -> pd.DataFrame:
        """Predict renewal rates and attach supporting model inputs."""

        renewal_model_key = "RENEWAL"
        renewal_res = self.renewal_model.predict(self._forecast_context)
        self._forecast_context.model_data[renewal_model_key] = renewal_res
        return renewal_res

    def calculate(self, calc_type: CalculationType) -> CalculationResult:
        """Execute the end-to-end calculation for *calc_type*."""

        calculated_data = {
            "OUTFLOW_PLAN": self.calculate_plan_outflows(),
            "COMPETITOR_RATES": self.calculate_competitors(),
            "MATURITY_STRUCTURE": self.calculate_maturity_structure(),
            "SIZE_STRUCTURE": self.calculate_size_structure(),
            "NEWBUSINESS": self.calculate_deposit_newbusiness(),
            "EARLY_REDEMPTION": self.calculate_early_redemption(),
            "RENEWAL": self.calculate_renewal(),
        }
        return CalculationResult(
            calc_type, self._scenario, {calc_type.name: calculated_data}
        )
