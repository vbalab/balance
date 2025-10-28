from __future__ import annotations
"""Tools that analyse calculator outputs and produce metrics or charts."""

from abc import ABC, abstractmethod
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd  # type: ignore[import-untyped]
from pandas.core.groupby.generic import (  # type: ignore[import-untyped]
    DataFrameGroupBy,
)
import plotly.graph_objects as go  # type: ignore[import-not-found]
from sklearn.metrics import (  # type: ignore[import-not-found]
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from core.upfm.commons import _REPORT_DT_COLUMN
from core.calculator.core import BackTestEngine

DEFAULT_DENOMINATOR = 10**9
DEFAULT_UNITS = "млрд. руб."
YAXIS_LIMITS: List[Optional[float]] = [None, None]
BASIC_METRICS = {
    "rmse": lambda x, y: mean_squared_error(x, y) ** 0.5,
    "mae": mean_absolute_error,
    "mape": mean_absolute_percentage_error,
    "r2": r2_score,
    "max_err": max_error,
}

MetricFunc = Callable[[pd.Series, pd.Series], float]


class CalculatorAnalyzer(ABC):
    """Abstract interface for calculator output analysers."""

    @abstractmethod
    def get_metrics(self, engine: BackTestEngine) -> Dict[Any, Any]:
        """Return evaluation metrics built from *engine* results."""

        pass

    @abstractmethod
    def get_chart_data(self, engine: BackTestEngine) -> Dict[Any, Any]:
        """Return chart-ready data frames built from *engine* results."""

        pass


class SimpleCalculatorAnalyzer(CalculatorAnalyzer):
    """Compare predictions and ground truth to produce metrics and charts."""

    denominator: int = DEFAULT_DENOMINATOR
    units: str = DEFAULT_UNITS

    def __init__(
        self,
        path_dict: Dict[Tuple[str, str], Tuple[str, str]],
        metrics: Dict[str, MetricFunc] = BASIC_METRICS,
    ) -> None:
        """Configure the analyser with model paths and metrics.

        Parameters
        ----------
        path_dict:
            Mapping from forecast result keys to ground truth paths. For
            example ``{('CurrentAccountsBalance', 'means_seas'): ('curr_acc_rur',
            'end_balance_amt')}``.
        metrics:
            Metric name to callable mapping. Defaults to :data:`BASIC_METRICS`.
        """

        self._metrics: Dict[str, MetricFunc] = metrics
        self._path_dict: Dict[Tuple[str, str], Tuple[str, str]] = path_dict

    def _fcst_extractor(
        self, engine: BackTestEngine, fcst_path: Tuple[str, str], backtest_step: int
    ) -> pd.DataFrame:
        """Retrieve prediction data for *fcst_path* at *backtest_step*."""

        return (
            engine.calc_results[backtest_step]
            .calculated_data[fcst_path[0]][[_REPORT_DT_COLUMN, fcst_path[1]]]
            .rename(columns={fcst_path[1]: "pred"})
        )

    def _ground_truth_extractor(
        self, engine: BackTestEngine, trth_path: Tuple[str, str], backtest_step: int
    ) -> pd.DataFrame:
        """Retrieve ground truth for *trth_path* at *backtest_step*."""

        return engine.ground_truth[(backtest_step, trth_path[0])]["target"][
            [_REPORT_DT_COLUMN, trth_path[1]]
        ].rename(columns={trth_path[1]: "truth"})

    def _get_product_name(self, fcst_path: Tuple[str, str]) -> str:
        """Return the product identifier for the provided *fcst_path*."""

        return fcst_path[0]

    def _get_chart_name(self, fcst_path: Tuple[str, str]) -> str:
        """Return a human readable chart name for *fcst_path*."""

        return "_".join(fcst_path)

    def _get_results_dict(
        self, engine: BackTestEngine
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Build aligned prediction and truth data for every configured path."""

        output: Dict[Tuple[str, str], pd.DataFrame] = {}

        for fcst_path, trth_path in self._path_dict.items():
            dfs: List[pd.DataFrame] = []

            for backtest_step in range(1, engine._config.steps + 1):
                fcst_df = self._fcst_extractor(engine, fcst_path, backtest_step)
                truth_df = self._ground_truth_extractor(
                    engine, trth_path, backtest_step
                )

                fcst_df = fcst_df.merge(truth_df, on=_REPORT_DT_COLUMN, how="inner")
                fcst_df["backtest_step"] = backtest_step
                fcst_df["product"] = self._get_product_name(fcst_path)
                fcst_df["train_last_date"] = (
                    pd.to_datetime(
                        engine.calc_results[backtest_step].config.forecast_dates[0]
                    )
                    - pd.offsets.MonthEnd()
                )
                fcst_df["periods_ahead"] = fcst_df[_REPORT_DT_COLUMN].dt.to_period(
                    "M"
                ).view(dtype="int64") - fcst_df["train_last_date"].dt.to_period(
                    "M"
                ).view(
                    dtype="int64"
                )
                fcst_df.loc[:, ["pred", "truth"]] = (
                    fcst_df.loc[:, ["pred", "truth"]] / self.denominator
                )

                dfs.append(fcst_df)

            output[fcst_path] = pd.concat(dfs, ignore_index=True)

        return output

    @staticmethod
    def _apply_metric(
        results_df_agg: DataFrameGroupBy,
        metric_name: str,
        metric_func: MetricFunc,
    ) -> pd.Series:
        """Apply a single metric to the grouped results."""

        return results_df_agg.apply(lambda df: metric_func(df.truth, df.pred)).rename(
            metric_name
        )

    def _apply_all_metrcis(
        self, results_df_agg: DataFrameGroupBy
    ) -> pd.DataFrame:
        """Apply all configured metrics to the provided aggregation."""

        output: List[pd.Series] = []
        for metric_name, metric_func in self._metrics.items():
            output.append(self._apply_metric(results_df_agg, metric_name, metric_func))
        return reduce(
            lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how="inner"),
            output,
        )

    def _get_metrics_one_df(self, results_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate step, horizon and overall metrics for *results_df*."""

        steps_metrics = self._apply_all_metrcis(results_df.groupby("backtest_step"))
        ahead_metrics = self._apply_all_metrcis(results_df.groupby("periods_ahead"))
        overall_metrcis = self._apply_all_metrcis(results_df.groupby("product"))

        return {
            "step_metrics": steps_metrics,
            "ahead_metrics": ahead_metrics,
            "overall_metrcis": overall_metrcis,
        }

    def get_metrics(
        self, engine: BackTestEngine
    ) -> Dict[Tuple[str, str], Dict[str, pd.DataFrame]]:
        """Return metrics for every configured forecast/target pair."""

        results_dict: Dict[Tuple[str, str], pd.DataFrame] = self._get_results_dict(
            engine
        )

        return {k: self._get_metrics_one_df(v) for k, v in results_dict.items()}

    def get_chart_data(
        self, engine: BackTestEngine
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Return raw chart data for every configured forecast/target pair."""

        nominal: Dict[Tuple[str, str], pd.DataFrame] = self._get_results_dict(engine)
        # for k, v in nominal.items():
        #    nominal[k].loc[:, ['pred', 'truth']] = v.loc[:, ['pred', 'truth']] / self.denominator

        return nominal

    def _plot_single_chart_data(
        self,
        chart_name: str,
        chart_data: pd.DataFrame,
        step_delimiters: bool,
        yaxis_limits: List[Optional[float]],
    ) -> None:
        """Render a single chart for *chart_name* with optional delimiters."""

        fig = go.Figure()
        # name = chart_data['product'].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=chart_data[_REPORT_DT_COLUMN],
                y=chart_data["truth"],
                # name = name+'_fact',
                name="fact",
                mode="lines+markers",
                line={"color": "blue"},
                # connectgaps=False,
            )
        )

        for step, step_df in chart_data.groupby("backtest_step"):
            fig.add_trace(
                go.Scatter(
                    x=step_df[_REPORT_DT_COLUMN],
                    y=step_df["pred"],
                    # name = name+'_pred',
                    name="pred",
                    mode="lines+markers",
                    line={"color": "red"},
                    showlegend=step == 1,
                )
            )

        # determine global y axis limits
        if yaxis_limits[0] is not None:
            global_min = min(chart_data[["truth", "pred"]].min().min(), yaxis_limits[0])
        else:
            global_min = chart_data[["truth", "pred"]].min().min()

        if yaxis_limits[1] is not None:
            global_max = max(chart_data[["truth", "pred"]].max().max(), yaxis_limits[1])
        else:
            global_max = chart_data[["truth", "pred"]].max().max()

        if step_delimiters:
            for step, step_df in chart_data.groupby("backtest_step"):
                if step > 1:
                    xdate = step_df["train_last_date"].iloc[0] + pd.offsets.Day(5)
                    fig.add_trace(
                        go.Scatter(
                            x=[xdate, xdate],
                            y=[global_min, global_max],
                            name=None,
                            mode="lines",
                            showlegend=False,
                            line={"color": "rgba(0,100,0,0.5)", "dash": "dash"},
                        )
                    )

        fig.update_layout(
            xaxis={
                "tickmode": "linear",
                "tick0": chart_data[_REPORT_DT_COLUMN].min(),
                "dtick": "M1",
            }
        )

        fig.update_layout(xaxis_tickformat="%b\n%Y")
        fig.update_layout(yaxis_title=self.units)
        fig.update_layout(title=chart_name)
        fig.update_yaxes(range=["auto" if v is None else v for v in yaxis_limits])

        fig.update_traces(connectgaps=False)

        fig.show()

    def plot_backtest(
        self,
        engine: BackTestEngine,
        step_delimiters: bool = True,
        yaxis_limits: List[Optional[float]] = YAXIS_LIMITS,
    ) -> None:
        """Render charts for every configured forecast/target pair."""

        chart_datas = self.get_chart_data(engine).items()

        for fcst_path, chart_data in chart_datas:
            chart_name = self._get_chart_name(fcst_path)

            self._plot_single_chart_data(
                chart_name,
                chart_data,
                step_delimiters,
                yaxis_limits,
            )


class SymbolicCalculatorAnalyzer(SimpleCalculatorAnalyzer):
    """Analyzer that evaluates symbolic expressions for predictions and truth."""

    def __init__(
        self,
        path_dict: Dict[Tuple[str, str], Tuple[str, str]],
        metrics: Optional[Dict[str, MetricFunc]] = None,
    ) -> None:
        """Initialise the analyser with optional custom metrics."""
        metrics_to_use = metrics if metrics is not None else BASIC_METRICS
        super().__init__(path_dict, metrics_to_use)

    def _get_results_dict(
        self, engine: BackTestEngine
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Build chart-ready data using symbolic expressions from *path_dict*."""

        output: Dict[Tuple[str, str], pd.DataFrame] = {}

        for fcst_path, trth_path in self._path_dict.items():
            dfs: List[pd.DataFrame] = []

            fcst_eval_str = f"""
            report_dt = report_dt
            pred = {fcst_path[1]}
            """
            trth_eval_str = f"""
            report_dt = report_dt
            truth = {trth_path[1]}
            """

            for backtest_step in range(1, engine._config.steps + 1):
                fcst_df = engine.calc_results[backtest_step].calculated_data[
                    fcst_path[0]
                ]
                fcst_df = fcst_df.eval(fcst_eval_str, engine="python")[
                    [_REPORT_DT_COLUMN, "pred"]
                ]

                truth_df = engine.ground_truth[(backtest_step, trth_path[0])]["target"]
                truth_df = truth_df.eval(trth_eval_str, engine="python")[
                    [_REPORT_DT_COLUMN, "truth"]
                ]

                fcst_df = fcst_df.merge(truth_df, on=_REPORT_DT_COLUMN, how="inner")
                fcst_df["backtest_step"] = backtest_step
                fcst_df["product"] = fcst_path[0]
                fcst_df["train_last_date"] = (
                    pd.to_datetime(
                        engine.calc_results[backtest_step].config.forecast_dates[0]
                    )
                    - pd.offsets.MonthEnd()
                )
                fcst_df["periods_ahead"] = fcst_df[_REPORT_DT_COLUMN].dt.to_period(
                    "M"
                ).view(dtype="int64") - fcst_df["train_last_date"].dt.to_period(
                    "M"
                ).view(
                    dtype="int64"
                )

                dfs.append(fcst_df)

            output[fcst_path] = pd.concat(dfs, ignore_index=True)

        return output
