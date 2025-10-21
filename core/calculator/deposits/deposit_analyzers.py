import logging
import pandas as pd
import plotly.express as px
from typing import Any, Dict
from pyspark.sql import SparkSession
from plotly.subplots import make_subplots

from core.upfm.commons import _REPORT_DT_COLUMN
from core.calculator.core import (
    AbstractEngine,
    BASIC_METRICS,
    CalculatorAnalyzer,
    Settings,
    SimpleCalculatorAnalyzer,
)

logging.config.dictConfig(Settings.LOGGING_CONFIG)
logger = logging.getLogger("core")


class DynbalanceCalculatorAnalyzer(CalculatorAnalyzer):
    """
    Основной класс аналитики прогнозов моделей депозитов ФЛ проекта динбаланс.

    Parameters
    ----------
    spark : spark-сессия: SparkSession = None
        Сессия для загрузки данных.
    metrics: Dict[str, Any] = BASIC_METRICS

    """

    def __init__(
        self,
        spark: SparkSession = None,
        metrics: Dict[str, Any] = BASIC_METRICS,
    ):
        self._spark = spark
        self._metrics = metrics

    def _fcst_extractor(self, engine: AbstractEngine, fcst_path, backtest_step):
        return (
            engine.calc_results[backtest_step]
            .calculated_data[fcst_path[0]][[_REPORT_DT_COLUMN, fcst_path[1]]]
            .rename(columns={fcst_path[1]: "pred"})
        )

    def _ground_truth_extractor(self, engine: AbstractEngine, step: int, tag: str):
        return engine.ground_truth[step][tag]["target"]

    def _pred_values_extractor(self, engine: AbstractEngine, step: int, tag: str):
        if tag.startswith("deposit_earlyredemption"):
            model = engine.register.get_model(engine.trained_models[(step, tag)])
            target_col = "SER_d_cl"

            pred = (
                engine.calc_results[step]
                .calculated_data["Deposits"]["agg_data"]
                .query(
                    f"segment == {model.segment} and replenishable_flg == {model.replenishable_flg} and subtraction_flg == {model.subtraction_flg}"
                )
                .set_index(_REPORT_DT_COLUMN)
                .loc[:, ["early_withdrawal", "operations"]]
                .sum(axis=1)
                .to_frame()
                .groupby(_REPORT_DT_COLUMN)
                .sum()
                .rename(columns={0: target_col})
            )
            return pred

        target = engine._config.data_loaders[tag].target
        model_data = engine.calc_results[step].calculated_data.get("model_data")

        return model_data.loc[:, target]

    def _results_in_sample_extractor(self, engine: AbstractEngine, step: int, tag: str):
        """
        Возвращает результат работы модели на тренировочных данных (in sample)
        """
        if not self._spark:
            raise ValueError("SparkSession is not available")

        dataloader = engine._config.data_loaders[tag]

        if tag.startswith("deposit_earlyredemption"):
            train_df = dataloader.get_training_data(
                self._spark,
                dataloader.start_date,
                engine._config.train_ends[step - 1],
            )
            train_data = train_df["data"]
            model = engine.register.get_model(engine.trained_models[(step, tag)])

            merge_cols = ["gen_name", _REPORT_DT_COLUMN]
            target_col = "SER_d_cl"

            truth = train_data[[*merge_cols, target_col]]
            pred = model._make_prediction_in_sample(train_data)
            pred = pred.merge(train_data, on=merge_cols)
            pred.loc[:, "predictions"] = (
                pred.loc[:, "predictions"]
                * pred.loc[:, model.model_class_kwargs["weight_col"]]
            )
            truth, pred = [
                df.groupby(_REPORT_DT_COLUMN)
                .agg({agg_col: "sum"})
                .rename(columns={agg_col: target_col})
                for df, agg_col in zip([truth, pred], [target_col, "predictions"])
            ]

        else:
            training_data = dataloader.get_training_data(
                self._spark,
                dataloader.default_start_date,
                engine._config.train_ends[step - 1],
            )
            model = engine.register.get_model(engine.trained_models[(step, tag)]).model

            truth = training_data["target"]
            pred = model.predict(training_data["features"])

        return truth, pred

    def _get_model_ground_truth(self, engine: AbstractEngine, tag):
        return pd.concat(
            [
                self._ground_truth_extractor(
                    engine, step, tag
                )  # .insert(0, 'backtest_step', step) \
                for step in range(1, engine._config.steps + 1)
            ]
        )

    def _get_model_pred_values(self, engine: AbstractEngine, tag):
        return pd.concat(
            [
                self._pred_values_extractor(
                    engine, step, tag
                )  # .insert(0, 'backtest_step', step) \
                for step in range(1, engine._config.steps + 1)
            ]
        )

    def _apply_metric(self, pred, truth, metric_name):
        try:
            return self._metrics[metric_name](pred, truth)
        except:
            logger.info(
                f"Metric `{metric_name}` does not support this input format of data"
            )

    def _apply_metrics(self, pred, truth) -> Dict[str, Any]:
        return {
            metric_name: self._apply_metric(pred, truth, metric_name)
            for metric_name in self._metrics.keys()
        }

    def get_model_metrics(self, engine: AbstractEngine, tag: str):
        metrics_data = []
        pred_all_steps = []
        truth_all_steps = []

        num_steps = engine._config.steps

        for step in range(1, num_steps + 1):
            pred = self._pred_values_extractor(engine, step, tag)
            truth = self._ground_truth_extractor(engine, step, tag)

            pred_all_steps.append(pred)
            truth_all_steps.append(truth)

            step_row = {
                "backtest_step": step,
                "first_forecast_date": engine._config.forecast_dates[step][0],
                "last_forecast_date": engine._config.forecast_dates[step][1],
                **self._apply_metrics(pred, truth),
            }

            metrics_data.append(step_row)

        metrics_data.append(
            {
                "backtest_step": "all",
                "first_forecast_date": engine._config.forecast_dates[1][0],
                "last_forecast_date": engine._config.forecast_dates[num_steps][1],
                **self._apply_metrics(
                    pd.concat(pred_all_steps), pd.concat(truth_all_steps)
                ),
            }
        )

        return pd.DataFrame(metrics_data)

    def get_metrics(self, engine: AbstractEngine):
        raise NotImplementedError("Use `get_model_metrics`")

    def _get_product_name(self, fcst_path, trth_path):
        return fcst_path[0]

    def _get_chart_name(self, fcst_path, trth_path):
        return "_".join(fcst_path)

    @staticmethod
    def _plot_single_chart_data(data, plot_method="line", **kwargs):
        fig = getattr(px, plot_method)(data, **kwargs)

        return fig

    def get_chart_data(self, engine: AbstractEngine, tag: str):
        num_steps = engine._config.steps
        pred = pd.concat(
            [
                self._pred_values_extractor(engine, step, tag)
                for step in range(1, num_steps + 1)
            ]
        )

        if len(pred.columns) > 1:
            return self._plot_single_chart_data(
                data=pred,
                plot_method="area",
                markers=True,
            )
        else:
            return self._plot_single_chart_data(data=pred, markers=True)

    def plot_backtest(
        self,
        engineengine: AbstractEngine,
        tag: str,
        add_in_sample=False,
        vertical_spacing=0.15,
        height=300,
        **kwargs,
    ):
        if add_in_sample:
            fig_list = []
            fig = make_subplots(
                rows=engine._config.steps,
                cols=1,
                vertical_spacing=(
                    vertical_spacing / (engine._config.steps - 1)
                    if engine._config.steps > 1
                    else 0
                ),
                subplot_titles=[
                    f"Backtest step {step}"
                    for step in range(1, engine._config.steps + 1)
                ],
            )

            for step in range(1, engine._config.steps + 1):
                truth_in_sample, pred_in_sample = self._results_in_sample_extractor(
                    engine, step, tag
                )
                truth = self._ground_truth_extractor(engine, step, tag)
                pred = self._pred_values_extractor(engine, step, tag)

                truth = pd.concat([truth_in_sample, truth], axis=0)
                pred = pd.concat(
                    [
                        df.assign(in_sample_flg=flg)
                        for flg, df in zip([1, 0], [pred_in_sample, pred])
                    ],
                    axis=0,
                )

                columns = truth.columns
                plot_df = pd.concat(
                    [
                        df.rename(columns={col: f"{col}{suffix}" for col in columns})
                        for suffix, df in zip(["_truth", "_pred"], [truth, pred])
                    ],
                    axis=1,
                )

                if len(columns) > 1:
                    fig = make_subplots(
                        rows=len(columns),
                        subplot_titles=columns,
                        vertical_spacing=vertical_spacing / (len(columns) - 1),
                    )
                    for row, col in enumerate(columns):
                        single_fig = self._plot_single_chart_data(
                            data=plot_df,
                            y=[c for c in plot_df.columns if c.startswith(col)],
                            markers=True,
                            line_group="in_sample_flg",
                        )
                        for trace in single_fig.data:
                            fig.append_trace(trace, row=row + 1, col=1)

                        fig.add_vline(
                            x=engine._config.train_ends[step - 1],
                            row="all",
                            col=1,
                            line_dash="dash",
                        )

                    fig.update_layout(
                        title=f"Backtest step {step} {tag}",
                        height=height * len(columns),
                        **kwargs,
                    )
                    fig_list.append(fig)
                else:
                    plot_df = plot_df.rename(
                        columns={
                            col: f"{col}_step_{step}"
                            for col in plot_df.columns
                            if col != "in_sample_flg"
                        }
                    )
                    single_fig = self._plot_single_chart_data(
                        data=plot_df,
                        y=[c for c in plot_df.columns if c.startswith(columns[0])],
                        markers=True,
                        line_group="in_sample_flg",
                    )
                    single_fig.add_vline(
                        x=engine._config.train_ends[step - 1], line_dash="dash"
                    )

                    for trace in single_fig.data:
                        fig.append_trace(trace, row=step, col=1)
                    fig.add_vline(
                        x=engine._config.train_ends[step - 1],
                        row=step,
                        col=1,
                        line_dash="dash",
                    )

            if len(fig_list) > 0:
                [fig.show() for fig in fig_list]
                return fig_list
            else:
                fig.update_layout(
                    title=f"{tag} Backtest",
                    height=height * (engine._config.steps + 1),
                    **kwargs,
                )
                return fig

        else:
            truth = self._get_model_ground_truth(engine, tag)

            pred = self._get_model_pred_values(engine, tag)
            columns = pred.columns
            pred = pred.rename(columns={col: f"{col}_pred" for col in pred.columns})
            truth = self._get_model_ground_truth(engine, tag)
            truth = truth.rename(columns={col: f"{col}_truth" for col in truth.columns})
            plot_df = pd.concat([pred, truth], axis=1)
            plot_df.insert(
                0,
                "backtest_step",
                [
                    step + 1
                    for step in range(engine._config.steps)
                    for h in range(engine._config.horizon)
                ],
            )

            fig = make_subplots(
                rows=len(columns),
                subplot_titles=columns,
                vertical_spacing=(
                    vertical_spacing / (len(columns) - 1) if len(columns) > 1 else 0
                ),
            )
            for row, col in enumerate(columns):
                single_fig = self._plot_single_chart_data(
                    data=plot_df,
                    y=[c for c in plot_df.columns if c.startswith(col)],
                    markers=True,
                    line_group="backtest_step",
                )
                for trace in single_fig.data:
                    fig.append_trace(trace, row=row + 1, col=1)

            for vline in engine._config.train_ends:
                fig.add_vline(x=vline, row="all", col=1, line_dash="dash")

            fig = fig.update_layout(height=height * len(columns), **kwargs)

            return fig


# Неактуальные на данный момент классы
class DepositAnalyzer(SimpleCalculatorAnalyzer):
    def __init__(self):
        path_dict = {"Deposits": "Deposits"}
        super().__init__(path_dict)

    def _fcst_extractor(self, engine, fcst_col, backtest_step):
        df = engine._calc_results[backtest_step].calculated_data["RetailDeposits"][
            "RENEWAL"
        ]["final_portfolio"]
        df.loc[:, "report_dt"] = pd.to_datetime(df["report_dt"])

        df = df.groupby("report_dt")["total_generation"].sum().reset_index().iloc[1:]
        df.columns = ["report_dt", "pred"]
        return df

    def _ground_truth_extractor(self, engine, trth_col, backtest_step):
        noopt_df = (
            engine.ground_truth[
                (backtest_step, "deposit_earlyredemption_noopt_vip_novip_RUR")
            ]["target"]
            .groupby(["report_dt"])[["total_generation"]]
            .agg("sum")
        )

        opt_df = (
            engine.ground_truth[
                (backtest_step, "deposit_earlyredemption_opt_vip_novip_RUR")
            ]["target"]
            .groupby(["report_dt"])[["total_generation"]]
            .agg("sum")
        )

        df = (noopt_df + opt_df).reset_index()
        df.columns = ["report_dt", "truth"]
        return df

    def _get_product_name(self, fcst_col, trth_col):
        return "deposits_portfolio"

    def _get_chart_name(self, fcst_path, trth_path):
        return "Прогноз портфеля депозитов"


class NewbusinessAnalyzer(SimpleCalculatorAnalyzer):
    def __init__(self):
        path_dict = {
            "NOVIP_NOOPT": "novip_noopt",
            "NOVIP_OPT": "novip_opt",
            "VIP_NOOPT": "vip_noopt",
            "VIP_OPT": "vip_opt",
        }
        super().__init__(path_dict)

    def _fcst_extractor(self, engine, fcst_path, backtest_step):
        df = (
            engine.calc_results[backtest_step]
            .calculated_data["RetailDeposits"]["NEWBUSINESS"][fcst_path]
            .reset_index()
        )
        df.columns = ["report_dt", "pred"]
        return df

    def _ground_truth_extractor(self, engine, trth_path, backtest_step):
        df = engine.ground_truth[(backtest_step, f"deposits_newbusiness_{trth_path}")][
            "target"
        ].reset_index()
        df.columns = ["report_dt", "truth"]
        return df

    def _get_product_name(self, fcst_path, trth_path):
        return f"deposits_newbusiness_{trth_path}"

    def _get_chart_name(self, fcst_path, trth_path):
        opt_dict = {"OPT": "опциональным", "NOOPT": "безопциональным"}
        # seg_dict = {'VIP': 'текущий портфель', 'NOVIP': 'новый бизнес'}
        return f"Прогноз нового бизнеса в сегменте {fcst_path.split('_')[0]} по {opt_dict[fcst_path.split('_')[1]]} вкладам"


class EarlyRedemptionAnalyzer(SimpleCalculatorAnalyzer):
    def __init__(self):
        path_dict = {
            ("NOOPT", "current_portfolio_agg"): (
                "deposit_earlyredemption_noopt_vip_novip_RUR",
                "newbiz_flag == False",
            ),
            ("OPT", "current_portfolio_agg"): (
                "deposit_earlyredemption_opt_vip_novip_RUR",
                "newbiz_flag == False",
            ),
            ("NOOPT", "newbiz_portfolio_agg"): (
                "deposit_earlyredemption_noopt_vip_novip_RUR",
                "newbiz_flag == True",
            ),
            ("OPT", "newbiz_portfolio_agg"): (
                "deposit_earlyredemption_opt_vip_novip_RUR",
                "newbiz_flag == True",
            ),
        }
        super().__init__(path_dict)

    def _fcst_extractor(self, engine, fcst_path, backtest_step):
        # [RetailDepositsCalculationType.RetailDeposits.name]['EARLY_REDEMPTION']['NOOPT']['newbiz_portfolio']

        return (
            engine.calc_results[backtest_step]
            .calculated_data["RetailDeposits"]["EARLY_REDEMPTION"][fcst_path[0]][
                fcst_path[1]
            ][["report_dt", "SER_d_cl"]]
            .iloc[1:]
            .groupby("report_dt", as_index=False)
            .sum()
            .rename(columns={"SER_d_cl": "pred"})
        )

    def _ground_truth_extractor(self, engine, trth_path, backtest_step):
        # ('deposit_earlyredemption_noopt_vip_novip_RUR', 'report_dt', 'SER_d')
        return (
            engine.ground_truth[(backtest_step, trth_path[0])]["target"]
            .query(trth_path[1])[["report_dt", "SER_d_cl"]]
            .groupby("report_dt", as_index=False)
            .sum()
            .rename(columns={"SER_d_cl": "truth"})
        )

    def _get_product_name(self, fcst_path, trth_path):
        return f"{fcst_path[0]}_{fcst_path[1][:-4]}"

    def _get_chart_name(self, fcst_path, trth_path):
        opt_dict = {"OPT": "опциональным", "NOOPT": "безопциональным"}
        new_dict = {
            "current_portfolio_agg": "текущий портфель",
            "newbiz_portfolio_agg": "новый бизнес",
        }
        return f"Прогноз досрочного погашения по {opt_dict[fcst_path[0]]} вкладам, {new_dict[fcst_path[1]]}"


class ReactionAnalyzer(SimpleCalculatorAnalyzer):
    def __init__(self):
        path_dict = {"SBER_RATE": "sber_rate", "NO_SBER_RATE": "no_sber_rate"}
        self.denominator = 1
        self.units = "Ставка, %"
        super().__init__(path_dict)

    def _fcst_extractor(self, engine, fcst_path, backtest_step):
        return (
            engine.calc_results[backtest_step]
            .calculated_data["RetailDeposits"]["COMPETITOR_RATES"][
                ["report_dt", fcst_path]
            ]
            .rename(columns={fcst_path: "pred"})
        )

    def _ground_truth_extractor(self, engine, trth_path, backtest_step):
        return engine.ground_truth[(backtest_step, "deposits_reactionrur_sarimax_v1")][
            "target"
        ][["report_dt", trth_path]].rename(columns={trth_path: "truth"})

    def _get_product_name(self, fcst_col, trth_col):
        return f"{fcst_col[:-5]}_reaction"

    def _get_chart_name(self, fcst_path, trth_path):
        return f"Прогноз индекса ставок {fcst_path}"


class MaturityStructureAnalyzer(SimpleCalculatorAnalyzer):
    def __init__(self):
        path_dict = {
            "NOVIP_NOOPT": "novip_noopt",
            "NOVIP_OPT": "novip_opt",
            "VIP_NOOPT": "vip_noopt",
            "VIP_OPT": "vip_opt",
        }
        self.denominator = 1
        self.units = "Cрочность, мес."
        self.terms = [3, 6, 12, 18, 24, 36, 37]
        super().__init__(path_dict)

    def _fcst_extractor(self, engine, fcst_path, backtest_step):
        df = engine.calc_results[backtest_step].calculated_data["RetailDeposits"][
            "MATURITY_STRUCTURE"
        ][fcst_path]
        weighted_maturity = sum(
            [df[df.columns[i]] * self.terms[i] for i in range(len(self.terms))]
        ).reset_index()
        weighted_maturity.columns = ["report_dt", "pred"]
        return weighted_maturity

    def _ground_truth_extractor(self, engine, trth_path, backtest_step):
        df = engine.ground_truth[
            (backtest_step, f"deposits_maturity_structure_{trth_path}")
        ]["target"].fillna(0)
        weighted_maturity = sum(
            [df[df.columns[i]] * self.terms[i] for i in range(len(self.terms))]
        ).reset_index()
        weighted_maturity.columns = ["report_dt", "truth"]
        return weighted_maturity

    def _get_product_name(self, fcst_path, trth_path):
        return f"deposits_maturity_structure_{trth_path}"

    def _get_chart_name(self, fcst_path, trth_path):
        opt_dict = {"OPT": "опциональным", "NOOPT": "безопциональным"}
        # seg_dict = {'VIP': 'текущий портфель', 'NOVIP': 'новый бизнес'}
        return f"Прогноз средневзвешенной срочности нового бизнеса в сегменте {fcst_path.split('_')[0]} по {opt_dict[fcst_path.split('_')[1]]} вкладам"


class DepositComponentAnalyzer(SimpleCalculatorAnalyzer):
    def __init__(self):
        path_dict = {
            "NOVIP_NOOPT": "NOVIP_NOOPT",
            "NOVIP_OPT": "NOVIP_OPT",
            "VIP_NOOPT": "VIP_NOOPT",
            "VIP_OPT": "VIP_OPT",
        }
        self.vip_map = {"VIP": 1, "NOVIP": 0}
        self.opt_map = {"OPT": ">0", "NOOPT": "==0"}
        super().__init__(path_dict)

    def _fcst_extractor(self, engine, fcst_path, backtest_step):
        vip, opt = fcst_path.split("_")
        df = (
            engine._calc_results[backtest_step]
            .calculated_data["RetailDeposits"]["RENEWAL"]["final_portfolio"]
            .query(f"vip_flg=={self.vip_map[vip]}")
            .query(f"optional_flg{self.opt_map[opt]}")
        )
        df.loc[:, "report_dt"] = pd.to_datetime(df["report_dt"])

        df = df.groupby("report_dt")["total_generation"].sum().reset_index().iloc[1:]
        df.columns = ["report_dt", "pred"]
        return df

    def _ground_truth_extractor(self, engine, trth_path, backtest_step):
        vip, opt = trth_path.split("_")
        if opt == "NOOPT":
            df = (
                engine.ground_truth[
                    (backtest_step, "deposit_earlyredemption_noopt_vip_novip_RUR")
                ]["target"]
                .query(f"vip_flg=={self.vip_map[vip]}")
                .query(f"optional_flg{self.opt_map[opt]}")
                .groupby(["report_dt"])[["total_generation"]]
                .agg("sum")
                .reset_index()
            )
        else:
            df = (
                engine.ground_truth[
                    (backtest_step, "deposit_earlyredemption_opt_vip_novip_RUR")
                ]["target"]
                .query(f"vip_flg=={self.vip_map[vip]}")
                .query(f"optional_flg{self.opt_map[opt]}")
                .groupby(["report_dt"])[["total_generation"]]
                .agg("sum")
                .reset_index()
            )
        # print(opt_df)
        # print(noopt_df)

        # df = (noopt_df + opt_df).reset_index()
        # print(df)
        df.columns = ["report_dt", "truth"]
        return df

    def _get_product_name(self, fcst_path, trth_col):
        return f"deposits_portfolio_{fcst_path.lower()}"

    def _get_chart_name(self, fcst_path, trth_path):
        return f"Прогноз портфеля депозитов в сегменте {fcst_path}"
