import numpy as np
import pandas as pd
from enum import auto
from typing import Dict
from functools import reduce
from datetime import datetime
from pandas.tseries.offsets import MonthEnd

from core.upfm.commons import (
    ModelInfo,
    BaseModel,
    Scenario,
    _REPORT_DT_COLUMN,
)
from core.calculator.core.calc_base import (
    ModelRegister,
    CalculationType,
    AbstractCalculator,
    CalculationResult,
)
from core.models import (
    PlanClose,
    MaturityStructure,
    OptStructure,
    Newbusiness,
    Renewal,
    EarlyWithdrawal,
    SaModels,
    CurrentAccounts,
    NewbusinessBuckets,
)
from core.models.utils import (
    get_feature_name,
    get_sa_feature_name,
    calculate_weighted_rates,
    calculate_absolute_inflows_nondefault_segments,
    parse_buckets_from_port,
)
from core.definitions import (
    OPTIONALS_,
    DEFAULT_SEGMENTS_,
    NONDEFAULT_SEGMENTS_,
    BUCKETED_BALANCE_MAP_,
    SEGMENT_MAP_,
    SA_PRODUCTS_,
)
from core.definitions import *


portfolio_result_cols = [
    "report_dt",
    "segment",
    "replenishable_flg",
    "subtraction_flg",
    "month_maturity",
    "target_maturity_days",
    "bucketed_balance_nm",
    "bucketed_balance",
    "open_month",
    "close_month",
    "weight_rate",
    "balance",
    "renewal_cnt",
    "operations_in_month",
    "early_withdrawal_in_month",
    "gen_name",
    "bucket_balance_share_[mass]_[0_500k)",
    "bucket_balance_share_[mass]_[500k_1500k)",
    "bucket_balance_share_[mass]_[1500k_5000k)",
    "bucket_balance_share_[mass]_[5000k_15000k)",
    "bucket_balance_share_[mass]_[15000k_inf)",
    "bucket_balance_share_[priv]_[0_500k)",
    "bucket_balance_share_[priv]_[500k_1500k)",
    "bucket_balance_share_[priv]_[1500k_5000k)",
    "bucket_balance_share_[priv]_[5000k_15000k)",
    "bucket_balance_share_[priv]_[15000k_inf)",
    "bucket_balance_share_[vip]_[0_15kk)",
    "bucket_balance_share_[vip]_[15kk_30kk)",
    "bucket_balance_share_[vip]_[30kk_50kk)",
    "bucket_balance_share_[vip]_[50kk_100kk)",
    "bucket_balance_share_[vip]_[100kk_200kk)",
    "bucket_balance_share_[vip]_[200kk_300kk)",
    "bucket_balance_share_[vip]_[300kk_500kk)",
    "bucket_balance_share_[vip]_[500kk_inf)",
    "null_cl_share",
    "pensioner_cl_share",
    "salary_cl_share",
    "standart_cl_share",
]


GROUP_AGG_COLS = [
    "segment",
    "replenishable_flg",
    "subtraction_flg",
    "target_maturity_days",
]


EW_AGG_COLS = ["report_dt", "segment", "open_month", "close_month"]


agg_right_order_cols = [
    "report_dt",
    "segment",
    "replenishable_flg",
    "subtraction_flg",
    "month_maturity",
    "target_maturity_days",
    "start_balance",
    "balance_gain",
    "balance",
    "newbusiness",
    "contract_close",
    "early_withdrawal",
    "operations",
    "interests",
    "renewal",
]

mat_right_order_cols = [
    "start_dttm",
    "end_dttm",
    "currency_id",
    "gbl_nm",
    "cf_cd",
    "cf_rate_type",
    "universal_weight_id",
    "weight_value",
]

volumes_right_order_cols = [
    "start_dttm",
    "end_dttm",
    "currency_id",
    "gbl_nm",
    "cf_cd",
    "cf_rate_type",
    "cf_new_issues_type",
    "cf_weight_distribution_type",
    "cf_volume_issues_type",
    "cf_value",
    "cf_value_sdo",
]

JOIN_COLS = [
    "report_dt",
    "segment",
    "replenishable_flg",
    "subtraction_flg",
    "month_maturity",
    "target_maturity_days",
    "open_month",
    "close_month",
    "weight_rate",
]


LIMIT_EARLY_WITHDROW = -20 * 10**9
COEF_EW_CONV_NB = -0.95

mat_table_dict_replace = {
    90: 2,
    180: 3,
    365: 5,
    548: 6,  # примерно, в таблице такого биннига нет
    730: 6,
    1095: 7,
}


class DepositsCalculationType(CalculationType):
    Deposits = auto()
    SavingAccounts = auto()
    CurrentAccounts = auto()


class DepositIterativeCalculator(AbstractCalculator):
    def __init__(
        self,
        model_register: ModelRegister,
        models: Dict[str, ModelInfo],
        scenario: Scenario,
        model_data: Dict[str, pd.DataFrame] = None,
    ) -> None:
        super().__init__(model_register, models, scenario, model_data)

        self._set_renewal_model()
        self._set_plan_close_model()
        self._set_balance_structure_models()
        self._set_maturity_structure_models()
        self._set_opt_structure_models()
        self._set_saving_accounts_models()
        self._set_newbusiness_models()
        self._set_early_withdrawal_models()
        self._set_current_accounts_models()

    def _set_renewal_model(self):
        self.renewal_model: BaseModel = self._model_register.get_model(
            self._models[Renewal.model_name]
        )

    def _set_plan_close_model(self):
        self.plan_close_model: BaseModel = self._model_register.get_model(
            self._models[PlanClose.model_name]
        )

    # Переписать сеттеры моделей - много дублирования кода (может как-то через дескрипторы сделать??)
    def _set_maturity_structure_model_by_params(
        self, segment: str, repl: int, sub: int
    ):
        model_name: str = MaturityStructure.get_model_by_conditions(
            segment=segment, replenishable_flg=repl, subtraction_flg=sub
        ).model_name

        model_info: ModelInfo = self._models[model_name]
        self.maturity_structure_models[(segment, repl, sub)] = (
            self._model_register.get_model(model_info)
        )

    def _set_early_withdrawal_model_by_params(self, segment: str, repl: int, sub: int):
        model_name: str = EarlyWithdrawal.get_model_by_conditions(
            segment=segment, replenishable_flg=repl, subtraction_flg=sub
        ).model_name

        model_info: ModelInfo = self._models[model_name]
        self.early_withdrawal_models[(segment, repl, sub)] = (
            self._model_register.get_model(model_info)
        )

    def _set_early_withdrawal_models(self):
        self.early_withdrawal_models: Dict[Tuple[str, int, int], BaseModel] = {}
        for segment in DEFAULT_SEGMENTS_:
            for repl, sub in OPTIONALS_:
                self._set_early_withdrawal_model_by_params(segment, repl, sub)

    def _set_maturity_structure_models(self):
        self.maturity_structure_models: Dict[Tuple[str, int, int], BaseModel] = {}
        for segment in DEFAULT_SEGMENTS_:
            for repl, sub in OPTIONALS_:
                self._set_maturity_structure_model_by_params(segment, repl, sub)

    def _set_balance_structure_models(self):
        self.balance_structure_models: Dict[Tuple[str, int, int], BaseModel] = {}
        for segment in DEFAULT_SEGMENTS_:
            model_name: str = NewbusinessBuckets.get_model_by_conditions(
                segment=segment
            ).model_name
            model_info: ModelInfo = self._models[model_name]
            self.balance_structure_models[segment] = self._model_register.get_model(
                model_info
            )

    def _set_opt_structure_models(self):
        self.opt_structure_models: Dict[str, BaseModel] = {}
        for segment in NONDEFAULT_SEGMENTS_:
            model_name: str = OptStructure.get_model_by_conditions(
                segment=segment
            ).model_name
            model_info: ModelInfo = self._models[model_name]
            self.opt_structure_models[segment] = self._model_register.get_model(
                model_info
            )

    def _set_newbusiness_models(self):
        self.newbusiness_models: Dict[str, BaseModel] = {}
        for segment in NONDEFAULT_SEGMENTS_:
            model_name: str = Newbusiness.get_model_by_conditions(
                segment=segment
            ).model_name
            model_info: ModelInfo = self._models[model_name]
            self.newbusiness_models[segment] = self._model_register.get_model(
                model_info
            )

    def _set_saving_accounts_models(self):
        self.saving_accounts_models: Dict[str, Dict[str, BaseModel]] = {}
        for sa_model in SaModels.models:
            model_name: str = sa_model.model_name
            model_info: ModelInfo = self._models[model_name]
            if not self.saving_accounts_models.get(sa_model.segment):
                self.saving_accounts_models[sa_model.segment] = {}
            self.saving_accounts_models[sa_model.segment][
                sa_model.model_trainer.prediction_type
            ] = self._model_register.get_model(model_info)

    def _set_current_accounts_models(self):
        self.current_accounts_models: Dict[str, BaseModel] = {}
        for ca_model in CurrentAccounts.models:
            model_name: str = ca_model.model_name
            model_info: ModelInfo = self._models[model_name]
            self.current_accounts_models[ca_model.model_trainer.prediction_type] = (
                self._model_register.get_model(model_info)
            )

    def _add_scenario_to_model_data(self, forecast_date):
        scenario_cols = self._scenario.scenario_data.columns
        self._forecast_context.model_data["features"].loc[
            forecast_date, scenario_cols
        ] = self._scenario.scenario_data.loc[forecast_date, :]

    def _add_predictions_to_model_data(
        self, prediction_df: pd.DataFrame, forecast_date: datetime
    ):
        cols = prediction_df.columns
        self._forecast_context.model_data["features"].loc[forecast_date, cols] = (
            prediction_df.loc[forecast_date, :]
        )

    def _add_weighted_rates(self, forecast_date, segment=None, repl=None, sub=None):
        df_date = self._forecast_context.model_data["features"].loc[[forecast_date], :]
        weighted_rate = calculate_weighted_rates(df_date, segment, repl, sub)
        feature_name = get_feature_name("VTB_weighted_rate", segment, repl, sub)
        self._forecast_context.model_data["features"].loc[
            [forecast_date], feature_name
        ] = weighted_rate

    def _calculate_absolute_inflows(self, forecast_date):
        df_date = self._forecast_context.model_data["features"].loc[[forecast_date], :]
        res = calculate_absolute_inflows_nondefault_segments(df_date)
        self._forecast_context.model_data["features"].loc[
            [forecast_date], res.columns
        ] = res

    def calculate_renewal(self, forecast_date):
        # При прогнозе в дату forecast_date берем портфель на предыдущую дату
        # (Предыдущая дата лежит в subcontext.portfolio_dt)
        # Делаем прогноз пролонгаций и кладем портфель со столбцом прогнозов обратно
        # По сути все манипуляции модели пролонгаций = взять портфель добавить столбец с прогнозом и положить обратно
        subcontext = self._forecast_context.subcontext(forecast_date)
        portf_with_renewals = self.renewal_model.predict(subcontext)

        self._forecast_context.model_data["portfolio"][
            subcontext.portfolio_dt
        ] = portf_with_renewals

    def calculate_plan_close(self, forecast_date) -> pd.DataFrame:
        outflows: pd.DataFrame = self.plan_close_model.predict(
            self._forecast_context.subcontext(forecast_date)
        )
        self._add_predictions_to_model_data(outflows, forecast_date)
        return outflows

    def calculate_balance_structure(self, forecast_date):
        for segment in DEFAULT_SEGMENTS_:
            model_ = self.balance_structure_models[segment]
            balance_structure = model_.predict(
                self._forecast_context.subcontext(forecast_date)
            )
            self._add_predictions_to_model_data(balance_structure, forecast_date)

    def calculate_maturity_structure(self, forecast_date):
        for segment in DEFAULT_SEGMENTS_:
            for repl, sub in OPTIONALS_:
                model_ = self.maturity_structure_models[(segment, repl, sub)]
                mat_structure_pred = model_.predict(
                    self._forecast_context.subcontext(forecast_date)
                )
                self._add_predictions_to_model_data(mat_structure_pred, forecast_date)
                self._add_weighted_rates(forecast_date, segment, repl, sub)

    def calculate_opt_structure(self, forecast_date):
        for segment in NONDEFAULT_SEGMENTS_:
            model_ = self.opt_structure_models[segment]
            opt_structure_pred = model_.predict(
                self._forecast_context.subcontext(forecast_date)
            )
            self._add_predictions_to_model_data(opt_structure_pred, forecast_date)
            if segment in ["mass", "priv"]:
                self._add_weighted_rates(forecast_date, segment)
        self._add_weighted_rates(forecast_date, segment="vip")

    def calculate_newbusiness(self, forecast_date):
        for segment in NONDEFAULT_SEGMENTS_:
            model_ = self.newbusiness_models[segment]
            newbusiness_pred = model_.predict(
                self._forecast_context.subcontext(forecast_date)
            )
            self._add_predictions_to_model_data(newbusiness_pred, forecast_date)
        self._calculate_absolute_inflows(forecast_date)

    def calculate_early_withdrawal(self, forecast_date):
        pred_res = []

        # сабконтекст передает признаки, даты, портфель и тд
        subcontext = self._forecast_context.subcontext(forecast_date)

        for segment in DEFAULT_SEGMENTS_:
            for repl, sub in OPTIONALS_:
                model_ = self.early_withdrawal_models[(segment, repl, sub)]
                portfolio_pred = model_.predict(subcontext)
                pred_res.append(portfolio_pred)
        portfolio_res = pd.concat(pred_res).reset_index(drop=True)
        self._forecast_context.model_data["portfolio"][forecast_date] = portfolio_res

    def calculate_saving_accounts_balance(self, forecast_date):
        for segment in DEFAULT_SEGMENTS_:
            segment_models = self.saving_accounts_models[segment]
            if segment_models.keys() == {"general_avg_balance", "kopilka_avg_balance"}:
                #                 kopilka_model_ = segment_models['kopilka_avg_balance']
                #                 kopilka_pred = kopilka_model_.predict(self._forecast_context.subcontext(forecast_date))
                general_model_ = segment_models["general_avg_balance"]
                general_pred = general_model_.predict(
                    self._forecast_context.subcontext(forecast_date)
                )

                #                 safe_pred = pd.DataFrame(data = general_pred.values - kopilka_pred.values,
                #                                          index = [forecast_date], columns = [f'SA_avg_balance_[safe]_[{segment}]'])
                sa_segment_balance_pred = general_pred.copy()

                sa_segment_balance_pred.index = [forecast_date]

                self._add_predictions_to_model_data(
                    sa_segment_balance_pred, forecast_date
                )

    def calculate_current_accounts_balance(self, forecast_date):
        curr_acc_models = self.current_accounts_models
        subcontext = self._forecast_context.subcontext(forecast_date)
        if curr_acc_models.keys() == {"general_avg_balance", "segment_structure"}:
            general_pred = curr_acc_models["general_avg_balance"].predict(subcontext)
            segment_shares_pred = curr_acc_models["segment_structure"].predict(
                subcontext
            )
            segment_balance_pred = pd.DataFrame(
                data=general_pred.values * segment_shares_pred.values,
                index=[forecast_date],
                columns=[
                    col.replace("CA_segment_share", "balance_rur")
                    for col in segment_shares_pred.columns
                ],
            )
            self._add_predictions_to_model_data(general_pred, forecast_date)
            self._add_predictions_to_model_data(segment_balance_pred, forecast_date)

    def run_step(self, forecast_date):
        self._add_scenario_to_model_data(forecast_date)

        self.calculate_renewal(forecast_date)
        self.calculate_plan_close(forecast_date)
        self.calculate_balance_structure(forecast_date)
        self.calculate_maturity_structure(forecast_date)
        self.calculate_opt_structure(forecast_date)
        self.calculate_saving_accounts_balance(forecast_date)
        self.calculate_newbusiness(forecast_date)
        self.calculate_early_withdrawal(forecast_date)
        self.calculate_current_accounts_balance(forecast_date)

    def run_expected_amount(self, step, forecast_date):
        """
        Добавление на первом шаге прогноза калибровочного коэффициента к новому бизнесу чтобы прогноз совпадал с ожидаемым значением.
        Ожидаемое значение задается в сценарии.
        """
        if (
            step == 0
            and (self._scenario.scenario_data.get("expected_amount") is not None)
            and (~np.isnan(self._scenario.scenario_data["expected_amount"].iloc[0]))
        ):  # шаг первый прогнозный и не ноль
            port_first = self._forecast_context.model_data["portfolio"][
                forecast_date
            ].copy()
            expected_amount = self._scenario.scenario_data["expected_amount"].iloc[
                0
            ]  # ожидаемое значение
            balance_pred = (
                port_first.groupby("report_dt").sum()["total_generation"].sum()
            )
            tmp_port = self._add_cols_to_port(port_first)
            newbiz = self._agg_newbusiness(tmp_port, GROUP_AGG_COLS)
            newbiz = newbiz["newbusiness"].sum()
            coef = (expected_amount - balance_pred + newbiz) / newbiz
            # отнормируем баланс
            cond = (port_first["report_month"] == port_first["open_month"]) & (
                port_first["weight_renewal_cnt"] < 1
            )
            self._forecast_context.model_data["portfolio"][forecast_date][
                "total_generation"
            ][cond] = (
                self._forecast_context.model_data["portfolio"][forecast_date][
                    "total_generation"
                ][cond]
                * coef
            )
            print(
                "Баланс сведен к заданному при помощи увеличения новых открытий на коэффициент ",
                coef,
            )

    def run_early_withdrawal_correct(self, step, forecast_date):
        """
        Корректируем перетоки из досрочного отзыва и операций в новый бизнес
        """
        port_first = self._forecast_context.model_data["portfolio"][
            forecast_date
        ].copy()

        if port_first["SER_d_cl"].sum() < LIMIT_EARLY_WITHDROW:
            # прибавляем к новому бизнесу перетоки
            # условие для выделения нового бизнеса
            cond = (port_first["report_month"] == port_first["open_month"]) & (
                port_first["weight_renewal_cnt"] < 1
            )

            # найдем распределение новго бизнеса и домножим н добавим коэффициенты

            weights = (
                self._forecast_context.model_data["portfolio"][forecast_date][
                    "total_generation"
                ][cond].abs()
            ) / self._forecast_context.model_data["portfolio"][forecast_date][
                "total_generation"
            ][
                cond
            ].abs().sum()

            self._forecast_context.model_data["portfolio"][forecast_date][
                "total_generation"
            ][cond] = (
                self._forecast_context.model_data["portfolio"][forecast_date][
                    "total_generation"
                ][cond]
                + weights * port_first["SER_d_cl"].sum() * COEF_EW_CONV_NB
            )

    def ew_share_calc(self, portfolio_res):
        port_ew = (
            portfolio_res.groupby(EW_AGG_COLS)
            .sum()[["balance", "early_withdrawal_in_month"]]
            .reset_index()
            .copy()
        )

        lag_data = port_ew.copy()
        lag_data.rename(columns={"balance": "balance_start1"}, inplace=True)
        lag_data["report_dt"] = lag_data.report_dt + MonthEnd(1)
        lag_data = lag_data[EW_AGG_COLS + ["balance_start1"]].copy()
        port_ew = port_ew.merge(lag_data, on=EW_AGG_COLS, how="outer")

        port_ew["share_ew"] = (
            port_ew["early_withdrawal_in_month"].abs() / port_ew["balance_start1"]
        )

        port_ew = port_ew[EW_AGG_COLS + ["share_ew"]]

        min_dt = port_ew.report_dt.min()
        max_dt = port_ew.report_dt.max()

        port_ew = port_ew[(port_ew["report_dt"] != min_dt)]
        port_ew = port_ew[(port_ew["report_dt"] != max_dt)]

        port_ew["report_dt"] = port_ew["report_dt"].dt.strftime("%d.%m.%Y")

        return port_ew

    def forecast_values_calc(self, agg_res1, CurrentAccounts1, SavingAccounts1):
        agg_res = agg_res1.copy()
        CurrentAccounts = CurrentAccounts1.copy()
        SavingAccounts = SavingAccounts1.copy()

        agg_res.rename(columns={"report_dt": "forecast_date"}, inplace=True)

        agg_res["product"] = "Deposits"

        cols = [
            "replenishable_flg",
            "subtraction_flg",
            "target_maturity_days",
            "start_balance",
            "balance_gain",
            "balance",
            "newbusiness",
            "contract_close",
            "early_withdrawal",
            "operations",
            "interests",
            "renewal",
            "forecast_date",
            "segment",
            "product",
        ]

        # CA
        CurrentAccounts = CurrentAccounts.reset_index().copy()
        CurrentAccounts.rename(columns={"report_dt": "forecast_date"}, inplace=True)
        CurrentAccounts["product"] = "CurrentAccounts"

        # kopilka
        SavingAccounts_koliplka = (
            SavingAccounts.reset_index()
            .rename(columns={"kopilka": "balance", "report_dt": "forecast_date"})
            .copy()
        )
        SavingAccounts_koliplka = SavingAccounts_koliplka[
            ["forecast_date", "segment", "balance"]
        ]

        SavingAccounts_koliplka["product"] = "SavingAccounts.kopilka"

        # safe
        SavingAccounts_safe = (
            SavingAccounts.reset_index()
            .rename(columns={"safe": "balance", "report_dt": "forecast_date"})
            .copy()
        )
        SavingAccounts_safe = SavingAccounts_safe[
            ["forecast_date", "segment", "balance"]
        ]

        SavingAccounts_safe["product"] = "SavingAccounts.safe"

        res = (
            agg_res.append(CurrentAccounts)
            .append(SavingAccounts_koliplka)
            .append(SavingAccounts_safe)[cols]
        )

        int_cols = ["replenishable_flg", "subtraction_flg", "target_maturity_days"]

        res[int_cols] = res[int_cols].astype("Int64")

        res["forecast_date"] = res["forecast_date"].dt.strftime("%d.%m.%Y")

        return res

    def _add_cols_to_port(self, port):
        port = port.copy()
        port.loc[:, "replenishable_flg"] = port["optional_flg"].isin([2, 3]).astype(int)
        port.loc[:, "subtraction_flg"] = port["optional_flg"].isin([1, 3]).astype(int)
        port.loc[:, "segment"] = port["is_vip_or_prv"].apply(lambda x: SEGMENT_MAP_[x])
        port.loc[:, "month_maturity"] = (port["bucketed_period"] - 1).astype(int)
        port.loc[:, "target_maturity_days"] = port.month_maturity.apply(
            month_to_target_maturity
        )
        return port

    def _add_orig_shares_balance(self, port):
        """
        Заполняем пропуски в оригинально портфеле
        Также проставляем нуллы для сегментов где доля ноль
        """
        # смотрим на оригинальный протфель
        portfolio_dt = self._forecast_context.portfolio_dt
        orig_port = port[port["report_dt"] == portfolio_dt]

        parse_res = {}

        parse_res["mass"] = parse_buckets_from_port(
            orig_port, segment="mass", balance_buckets=MASS_BALANCE_BUCKETS
        )
        parse_res["priv"] = parse_buckets_from_port(
            orig_port, segment="priv", balance_buckets=PRIV_BALANCE_BUCKETS
        )
        parse_res["vip"] = parse_buckets_from_port(
            orig_port, segment="vip", balance_buckets=VIP_BALANCE_BUCKETS
        )

        for segm in ["mass", "priv", "vip"]:
            parse_res_new = parse_res[segm]

            pred = pd.DataFrame(
                data=parse_res_new, index=[portfolio_dt.strftime("%Y-%m-%d")]
            ).add_prefix(f"bucket_balance_share_[{segm}]_")

            # заполняем оригинальный портфель - старые открытия вкладов
            port.loc[
                port["open_month"] <= portfolio_dt.strftime("%Y-%m"), pred.columns
            ] = pred.values

            # заполняем прологированные
            port.loc[
                (
                    (port["renewal_cnt"] >= 0)
                    & ((port[pred.columns].isna()).max(axis=1))
                ),
                pred.columns,
            ] = pred.values

            # Заполняем все остальные знаечния у сегментов нуллами
            port.loc[port["segment"] != segm, pred.columns] = 0

        return port

    def _add_orig_shares_client_types(self, port):
        """
        Функция заполняет доли по типам клиентов
        """

        # смотрим на оригинальный протфель
        portfolio_dt = self._forecast_context.portfolio_dt
        orig_port = port[port["report_dt"] == portfolio_dt]

        # выделяем максимальные значения по долям
        cols_clients = [
            "3_med_pr_null",
            "3_med_pr_PENSIONER",
            "3_med_pr_SALARY",
            "3_med_pr_STANDART",
        ]

        rename_dict = {
            "3_med_pr_null": "null_cl_share",
            "3_med_pr_PENSIONER": "pensioner_cl_share",
            "3_med_pr_SALARY": "salary_cl_share",
            "3_med_pr_STANDART": "standart_cl_share",
        }

        cl_values = orig_port.groupby("segment")[cols_clients].max().fillna(0)

        cl_values = cl_values.rename(columns=rename_dict)

        port = port.merge(cl_values, on="segment", how="left")

        return port

    def _portfolio_result(self, port):
        port["bucketed_balance_nm"] = port.bucketed_balance.apply(
            lambda x: (
                BUCKETED_BALANCE_MAP_[x] if x in BUCKETED_BALANCE_MAP_ else "<100k"
            )
        )
        port.loc[:, "renewal_cnt"] = port.weight_renewal_cnt.round()
        port.loc[:, "operations_in_month"] = np.where(
            port.optional_flg > 0, port.SER_d_cl, 0
        )
        port.loc[:, "early_withdrawal_in_month"] = np.where(
            port.optional_flg == 0, port.SER_d_cl, 0
        )
        port.loc[:, "balance"] = port.loc[:, "total_generation"]

        port = port[port["report_dt"] >= self._forecast_context.portfolio_dt]

        # сделаем заполнение бакетов баланса для оригинального портфеля
        port = self._add_orig_shares_balance(port)

        # заполняем доли по типам клиентов
        port = self._add_orig_shares_client_types(port)

        return port[portfolio_result_cols].sort_values(
            by=[
                "report_dt",
                "segment",
                "replenishable_flg",
                "subtraction_flg",
                "balance",
            ],
            ascending=[True, True, True, True, False],
            ignore_index=True,
        )

    def _agg_contract_close_fact(self, group_cols):
        res = pd.DataFrame()
        for date in self._forecast_context.forecast_dates:
            previous_dt = date + MonthEnd(-1)
            previos_port_dt = self._forecast_context.model_data["portfolio"][
                previous_dt
            ]
            close_port = previos_port_dt[
                previos_port_dt.close_month == date.strftime("%Y-%m")
            ]
            close_port = self._add_cols_to_port(close_port)
            agg_close = (
                close_port.groupby(["close_month"] + group_cols)[
                    ["total_generation", "renewal_balance_next_month"]
                ]
                .sum()
                .reset_index()
            )
            res = res.append(agg_close)
        res["contract_close"] = (
            res["total_generation"] - res["renewal_balance_next_month"]
        ) * -1
        res = res.rename(columns={"close_month": _REPORT_DT_COLUMN})
        res[_REPORT_DT_COLUMN] = pd.to_datetime(res[_REPORT_DT_COLUMN]) + MonthEnd(0)
        return res[[_REPORT_DT_COLUMN] + group_cols + ["contract_close"]]

    def _agg_renewal(self, port, group_cols):
        renewal = port.query(
            "(report_month == open_month) and (weight_renewal_cnt >= 1) "
        )

        renewal = (
            renewal.groupby([_REPORT_DT_COLUMN] + group_cols)["total_generation"]
            .sum()
            .reset_index()
        )
        renewal = renewal.rename(columns={"total_generation": "renewal"})
        return renewal

    def _agg_newbusiness(self, port, group_cols):
        newbiz = port.query(
            "(report_month == open_month) and (weight_renewal_cnt < 1) "
        )

        newbiz = (
            newbiz.groupby([_REPORT_DT_COLUMN, *group_cols])["total_generation"]
            .sum()
            .reset_index()
        )
        newbiz = newbiz.rename(columns={"total_generation": "newbusiness"})
        return newbiz

    def _agg_early_withdrawal(self, port, group_cols):
        ew = (
            port.groupby([_REPORT_DT_COLUMN, *group_cols])["SER_d_cl"]
            .sum()
            .reset_index()
        )
        ew["early_withdrawal"] = np.where(
            (ew.replenishable_flg > 0) | (ew.subtraction_flg > 0), 0, ew.SER_d_cl
        )
        ew["operations"] = np.where(
            (ew.replenishable_flg > 0) | (ew.subtraction_flg > 0), ew.SER_d_cl, 0
        )
        return ew[[_REPORT_DT_COLUMN, *group_cols, "early_withdrawal", "operations"]]

    def _agg_interests(self, port, group_cols):
        old_port = port[port.total_generation_lag1 > 0]
        old_port["interests"] = (
            old_port["total_generation"]
            - old_port["total_generation_lag1"].astype(float)
            - old_port["SER_d_cl"]
        )
        interests = (
            old_port.groupby([_REPORT_DT_COLUMN, *group_cols])["interests"]
            .sum()
            .reset_index()
        )
        return interests

    def _agg_balance(self, port, group_cols):
        bal = (
            port.groupby([_REPORT_DT_COLUMN, *group_cols])["total_generation"]
            .sum()
            .reset_index()
        )
        bal = bal.rename(columns={"total_generation": "balance"})
        return bal

    def _agg_saving_accounts_balance(self):
        model_data = self._forecast_context.model_data["features"]
        sa_balance = []
        for segment in DEFAULT_SEGMENTS_:
            segment_balance = pd.DataFrame(
                {"segment": [segment]}, index=model_data.index
            )
            for product in SA_PRODUCTS_:
                segment_balance.loc[:, product] = model_data.loc[
                    :, get_sa_feature_name("SA_avg_balance", product, segment)
                ]
            sa_balance.append(segment_balance)
        sa_balance_table = pd.concat(sa_balance, axis=0).reset_index()
        sa_balance_table.columns.name = "balance"
        return sa_balance_table

    def _agg_current_accounts_balance(self):
        model_data = self._forecast_context.model_data["features"]
        ca_balance = []
        for segment in [*DEFAULT_SEGMENTS_]:
            segment_balance = pd.DataFrame(
                data={
                    "segment": [segment],
                    "balance": model_data.loc[:, f"balance_rur_[{segment}]"],
                },
                index=model_data.index,
            )
            ca_balance.append(segment_balance)
        ca_balance = pd.concat(ca_balance, axis=0)
        return ca_balance

    def _aggregated_data(self, port, group_cols):
        contract_close = self._agg_contract_close_fact(group_cols)
        renewal = self._agg_renewal(port, group_cols)
        newbiz = self._agg_newbusiness(port, group_cols)
        ew_oper = self._agg_early_withdrawal(port, group_cols)
        interests = self._agg_interests(port, group_cols)
        balance = self._agg_balance(port, group_cols)
        res = reduce(
            lambda x, y: x.merge(y, on=[_REPORT_DT_COLUMN, *group_cols], how="outer"),
            [balance, newbiz, contract_close, ew_oper, interests, renewal],
        )
        return res.fillna(0)

    def _maturity_table(self, agg_df):
        """
        Создание таблицы по срочностям
        """
        agg_df["universal_weight_id"] = agg_df.target_maturity_days.replace(
            mat_table_dict_replace
        )
        # расчет долей
        agg_df = (
            agg_df.groupby(["report_dt", "universal_weight_id"]).sum()["balance"]
            / agg_df.groupby(["report_dt"]).sum()["balance"]
        ).reset_index()
        agg_df.rename(
            columns={"report_dt": "end_dttm", "balance": "weight_value"}, inplace=True
        )
        agg_df["start_dttm"] = agg_df["end_dttm"].apply(lambda x: x.replace(day=1))
        agg_df["weight_value"] = agg_df["weight_value"].round(2)
        agg_df["currency_id"] = "RUB"
        agg_df["gbl_nm"] = "РБ"
        agg_df["cf_cd"] = "B_L_DEP_IND_TERM_DEP"
        agg_df["cf_rate_type"] = "FIXED"

        return agg_df[mat_right_order_cols]

    def _volumes_table(self, agg_data, sa_data, ca_data):
        """
        Создание таблицы по объемам
        """
        agg_df = pd.DataFrame(agg_data.groupby("report_dt").sum()["balance"]).rename(
            columns={"balance": "cf_value_sdo"}
        )
        agg_df.loc[:, "cf_cd"] = "B_L_DEP_IND_TERM_DEP"

        delta = agg_df["cf_value_sdo"].diff(1).fillna(method="bfill") / 2
        delta[0] = delta[0]
        agg_df["cf_value"] = agg_df["cf_value_sdo"] + delta

        sa_df = pd.DataFrame(sa_data.groupby("report_dt").sum()["general"]).rename(
            columns={"general": "cf_value_sdo"}
        )
        sa_df.loc[:, "cf_cd"] = "B_L_DEP_IND_SAV_ACC"

        delta = sa_df["cf_value_sdo"].diff(1).fillna(method="bfill") / 2
        delta[0] = delta[0]
        sa_df["cf_value"] = sa_df["cf_value_sdo"] + delta

        ca_df = pd.DataFrame(ca_data.groupby("report_dt").sum()["balance"]).rename(
            columns={"balance": "cf_value_sdo"}
        )
        ca_df.loc[:, "cf_cd"] = "B_L_DEP_IND_CUR_ACC"

        delta = ca_df["cf_value_sdo"].diff(1).fillna(method="bfill") / 2
        delta[0] = delta[0]
        ca_df["cf_value"] = ca_df["cf_value_sdo"] + delta

        res = pd.concat([agg_df, sa_df, ca_df], axis=0).reset_index()
        res.rename(columns={"report_dt": "end_dttm"}, inplace=True)
        res["start_dttm"] = res["end_dttm"].apply(lambda x: x.replace(day=1))
        res["currency_id"] = "RUB"
        res["gbl_nm"] = "РБ"
        res["cf_rate_type"] = "FIXED"
        res["cf_new_issues_type"] = "interest_rate_new"
        res["cf_weight_distribution_type"] = "new_issue"
        res["cf_volume_issues_type"] = "target_issue"

        return res[volumes_right_order_cols]

    def _check_decomposition(self, macro, decomp_max_error=1e9):
        decomp_cols = [
            "newbusiness",
            "contract_close",
            "early_withdrawal",
            "operations",
            "interests",
        ]
        start_bal = (
            self._forecast_context.model_data["portfolio"][
                self._forecast_context.portfolio_dt
            ]["total_generation"]
            .sum()
            .astype(float)
        )
        decomp_bal_cumulative_difference = (
            macro.groupby(_REPORT_DT_COLUMN)[decomp_cols]
            .sum()
            .sum(axis=1)
            .cumsum()
            .astype(float)
        )
        decomp_theory_balance = start_bal + decomp_bal_cumulative_difference
        fact_bal = macro.groupby(_REPORT_DT_COLUMN)["balance"].sum().astype(float)
        decomp_error = decomp_theory_balance - fact_bal
        if any(decomp_error > decomp_max_error):
            raise ValueError(
                f"Decomposition doesn't converge. Fact balance like: {fact_bal}, decomposed_balance looks like: {decomp_theory_balance}"
            )

    def _add_start_balance(self, agg_data):
        """
        Добавляем стартовый баланс в агрегированную таблицу с депозитами
        2 этапа - на начальную дату берем из портфеля
        Далее считаем как лаг от предыдущего баланса
        группировка в разрезе по месяцам
        """
        # расчет значений
        lag_data = agg_data.copy()
        lag_data.rename(columns={"balance": "start_balance"}, inplace=True)
        lag_data["report_dt"] = lag_data.report_dt + MonthEnd(1)
        lag_data = lag_data[GROUP_AGG_COLS + ["report_dt", "start_balance"]].copy()
        agg_data = agg_data.merge(
            lag_data, on=GROUP_AGG_COLS + ["report_dt"], how="left"
        )

        agg_data = agg_data[agg_data["report_dt"] > self._forecast_context.portfolio_dt]

        # добавим прирост баланса
        decomp_cols = [
            "newbusiness",
            "contract_close",
            "early_withdrawal",
            "operations",
            "interests",
        ]
        agg_data["balance_gain"] = agg_data[decomp_cols].sum(axis=1)
        agg_data["month_maturity"] = agg_data.target_maturity_days.replace(
            MATURITY_TO_MONTH_MAP_
        )

        # делаем логичный вывод колонок
        agg_data = agg_data[agg_right_order_cols]

        # убираем портфель за предыдущий этап

        return agg_data

    def calculate(self, calc_type: CalculationType) -> CalculationResult:
        port_res = []
        # добавляем исходный портфель, потом его уберем
        port_res.append(
            self._forecast_context.model_data["portfolio"][
                self._forecast_context.portfolio_dt
            ]
        )
        for step, forecast_date in enumerate(self._scenario.forecast_dates):
            # прогоняем модели
            self.run_step(forecast_date)
            # корректируем переток
            self.run_early_withdrawal_correct(step, forecast_date)
            # корректировка на ожидаемые значения нового бизнеса
            self.run_expected_amount(step, forecast_date)
            port_res.append(
                self._forecast_context.model_data["portfolio"][forecast_date]
            )
        port = self._add_cols_to_port(pd.concat(port_res).reset_index(drop=True))
        agg_data = self._aggregated_data(port, GROUP_AGG_COLS)

        # здесь счиатем доли по бакетам и другие колонки
        portfolio_res = self._portfolio_result(port)

        # А здесь доюбавим агрегацию для

        # portfolio_res = portfolio_res[portfolio_res['report_dt']>self._forecast_context.portfolio_dt]
        # добавим вывод стартового баланса по месяцам агрегировано
        agg_data = self._add_start_balance(agg_data)
        # вывод таблицы со срочностями
        maturity_table = self._maturity_table(agg_data)
        # расчет коэффициента для досрочного отзыва
        ew_share_table = self.ew_share_calc(portfolio_res)

        calculated_data = {
            "portfolio": portfolio_res,
            "agg_data": agg_data,
            "maturity": maturity_table,
            "ew_share": ew_share_table,
        }

        self._check_decomposition(agg_data)

        sa_data = self._agg_saving_accounts_balance()
        ca_data = self._agg_current_accounts_balance()

        #         forecast_values = self.forecast_values_calc(agg_data, ca_data, sa_data)

        # вывод таблицы итоговых объемов
        #         volumes_table = self._volumes_table(agg_data, sa_data, ca_data)

        return CalculationResult(
            calc_type,
            self._scenario,
            {
                calc_type.Deposits.name: calculated_data,
                calc_type.SavingAccounts.name: sa_data,
                calc_type.CurrentAccounts.name: ca_data,
                #                                                             'Volumes': volumes_table,
                #                                                             'forecast_values': forecast_values,
                "model_data": self._forecast_context.model_data[
                    "features"
                ],  # Добавлен ключ model_data для работы DynbalanceCalculatorAnalyzer
            },
        )
