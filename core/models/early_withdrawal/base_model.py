import pandas as pd
import numpy as np
import statsmodels.api as sm

# from datetime import datetime
# import datetime as dt
# from datetime import timedelta
from itertools import product
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from core.upfm.commons import _REPORT_DT_COLUMN

# import pickle
from .PanelOLS import PanelOLS

# try:
#     from .PanelOLS import PanelOLS
# except ModuleNotFoundError:
#     from PanelOLS import PanelOLS


def early_redemption_ptr_transform(df, cols, labels, cols_tresholds, exog):
    # осторожно, эта функция добавляет константу независимо от параметра add_constant!
    # TODO: переделать так, чтобы можно было строить модель без константы
    if len(cols) > 0:
        tresholds = [[-np.inf] + x + [np.inf] for x in cols_tresholds]
        cols_tresh_list = []
        for i, col in enumerate(cols):
            col_tresh_list = []
            label = labels[i]
            for t in range(len(tresholds[i]) - 1):
                # df[f'{label}{t}'] = np.int64((tresholds[i][t+1] > df.reset_index()[col]) * (df.reset_index()[col] >= tresholds[i][t]))
                df[f"{label}{t}"] = (
                    (df.reset_index()[col].values < tresholds[i][t + 1])
                    & (df.reset_index()[col].values >= tresholds[i][t])
                ).astype("float")
                col_tresh_list.append(f"{label}{t}")
            cols_tresh_list.append(col_tresh_list)
        groups = [x for x in product(*cols_tresh_list)]
        exog_vars = []
        for group in groups:
            group_id = "".join(group)
            df[group_id] = df.loc[:, group].prod(axis=1)
            exog_vars.append(group_id)
            for col in exog:
                df[f"{group_id}*{col}"] = df[group_id] * df[col]
                exog_vars.append(f"{group_id}*{col}")
        # print('groups', [''.join(x) for x in groups])
        return df, exog_vars  # , [''.join(x) for x in groups]
    else:
        df["const"] = np.zeros(len(df)) + 1
        exog_vars = exog + ["const"]
        return df, exog_vars  # , []


# def early_redemption_ptr_transform(df, cols, labels, cols_tresholds, exog):

#     ch = df.copy()
#     tresholds = [[-np.inf] + x + [np.inf] for x in cols_tresholds]
#     cols_tresh_list = []
#     for i, col in enumerate(cols):
#         col_tresh_list = []
#         label = labels[i]
#         for t in range(len(tresholds[i])-1):
#             ch[f'{label}{t}'] = (
#                 np.int64(
#                     (tresholds[i][t+1] > ch.reset_index()[col])
#                     &(ch.reset_index()[col] >= tresholds[i][t])
#                 )
#             )
#             col_tresh_list.append(f'{label}{t}')
#         cols_tresh_list.append(col_tresh_list)
#     groups = [x for x in product(*cols_tresh_list)]
#     exog_vars = []
#     for group in groups:
#         group_id = ''.join(group)
#         ch[group_id] = ch.loc[:, group].prod(axis=1)
#         exog_vars.append(group_id)
#         for col in exog:
#             ch[f'{group_id}*{col}'] = ch[group_id] * ch[col]
#             exog_vars.append(f'{group_id}*{col}')
#     return ch, exog_vars


class EarlyRedemptionModel:
    def __init__(
        self,
        target,
        leaf_reg_features,
        splitting_cols,
        splitting_cols_labels,
        weight_col,
    ) -> None:
        self.target = target
        self.leaf_reg_features = leaf_reg_features
        self.splitting_cols = splitting_cols
        self.splitting_cols_labels = splitting_cols_labels
        self.weight_col = weight_col

        # if cur == 'RUR':
        #     self.n_iter = 2
        #     self.alpha = 2e-5
        # elif cur == 'USD':
        #     self.n_iter = 3
        #     self.alpha = 7e-6
        # self.cur = cur
        # self.date = date

    def lin_reg(
        self,
        train,
        features,
        target,
        weights,
        scaler,
        add_constant,
        lasso_alpha,
        print_summary,
        entity_effects,
        rank_check,
    ):
        if weights:
            train_temp = train.set_index(["gen_name", _REPORT_DT_COLUMN])[
                features + [target, weights]
            ].dropna()
            train_temp = train_temp[train_temp[weights] > 0]
        else:
            train_temp = train.set_index(["gen_name", _REPORT_DT_COLUMN])[
                features + [target]
            ].dropna()

        X_train = train_temp[features].astype(np.float64)
        y_train = train_temp[target].astype(np.float64)

        if scaler:
            sc = StandardScaler()
            sc.fit(X_train)
            self.sc = sc
            cols, ind = X_train.columns, X_train.index
            X_train = pd.DataFrame(sc.transform(X_train), index=ind, columns=cols)

        if add_constant:
            X_train = sm.add_constant(X_train)

        if lasso_alpha:
            if weights:
                reg = PanelOLS(
                    y_train,
                    X_train,
                    weights=train_temp[weights],
                    entity_effects=entity_effects,
                    check_rank=rank_check,
                ).fit(cov_type="robust", lasso_l1=lasso_alpha, rank_check=rank_check)
            else:
                reg = PanelOLS(
                    y_train,
                    X_train,
                    entity_effects=entity_effects,
                    check_rank=rank_check,
                ).fit(
                    cov_type="robust",
                    lasso_l1=lasso_alpha,
                    rank_check=rank_check,
                )

        else:
            if weights:
                reg = PanelOLS(
                    y_train,
                    X_train,
                    weights=train_temp[weights],
                    entity_effects=entity_effects,
                    check_rank=rank_check,
                ).fit(
                    cov_type="robust",
                    rank_check=rank_check,
                )
            else:
                reg = PanelOLS(
                    y_train,
                    X_train,
                    entity_effects=entity_effects,
                    check_rank=rank_check,
                ).fit(
                    cov_type="robust",
                    rank_check=rank_check,
                )

        if print_summary:
            print(reg)

        return reg

    # Вынели predict в отдельную функци
    def predict(self, reg, test_df, y_train, scaler, add_constant, significance=1):
        if reg is None:
            test_df["predict_reg"] = 0.0
            test_df["predict_mean"] = 0.0
            return test_df

        features = reg.pvalues[reg.pvalues < significance].index.tolist()
        X_test = test_df.set_index(["gen_name", _REPORT_DT_COLUMN])[features].astype(
            np.float64
        )
        if scaler:
            cols, ind = X_test.columns, X_test.index
            X_test = pd.DataFrame(self.sc.transform(X_test), index=ind, columns=cols)

        if add_constant:
            X_test = sm.add_constant(X_test)

        test_df["predict_reg"] = reg.predict(X_test).values
        test_df["predict_reg"] = np.minimum(test_df["predict_reg"], y_train.max())
        test_df["predict_reg"] = np.maximum(test_df["predict_reg"], -1)

        test_df["predict_mean"] = y_train[y_train < 1].mean()

        return test_df

    def forecast_balance(self, test, predictCols=["predict_reg", "predict_mean"]):
        last_gen_balance = test["total_generation_cl_lag1"].astype(np.float64)

        for col in predictCols:
            test[f"total_generation_cl_{col}"] = last_gen_balance * (
                1 + test[col].astype(np.float64)
            )

        return test

    def forecast_balance_long(
        self, test, months, start_date, plot=True, title=None, cur="RUB"
    ):
        test = test[test["open_month"] < start_date]

        res = []

        for i in range(1, months + 1):
            if i == 1:
                test["forecast"] = test["total_generation_cl_lag1"].astype(
                    np.float64
                ) * (1 + test["predict_reg"].astype(np.float64))
                test["forecast_contract"] = test["total_generation_cl_lag1"].astype(
                    np.float64
                )
                test["redemption"] = test["total_generation_cl_lag1"].astype(
                    np.float64
                ) * test["predict_reg"].astype(np.float64)
            else:
                test["forecast"] = test.groupby("gen_name")["forecast"].shift()
                test["forecast_contract"] = test.groupby("gen_name")[
                    "forecast_contract"
                ].shift()
                test["redemption"] = test["forecast"].astype(np.float64) * test[
                    "predict_reg"
                ].astype(np.float64)
                test["forecast"] = test["forecast"].astype(np.float64) * (
                    1 + test["predict_reg"].astype(np.float64)
                )

            cur_date = pd.to_datetime(start_date) + pd.offsets.MonthEnd(1 + i)

            dat_cur = test[test[_REPORT_DT_COLUMN] == cur_date]

            forecast = dat_cur.forecast.sum()

            redemption = dat_cur.redemption.sum()

            redemption_real = np.sum(
                dat_cur["total_generation_cl_lag1"] * dat_cur["SER_dinamic_cl"]
            )

            contract = dat_cur.forecast_contract.sum()

            real = dat_cur.total_generation_cleared.sum()

            res.append(
                (cur_date, forecast, contract, real, redemption, redemption_real)
            )

        res = pd.DataFrame(
            res,
            columns=[
                "Дата",
                "Прогноз модели",
                "Прогноз по контрактному погашению",
                "Фактические значения",
                "Прогноз досрочных погашений",
                "Досрочные погашения",
            ],
        )

        res["Прогноз модели"] = res[
            ["Прогноз модели", "Прогноз по контрактному погашению"]
        ].min(axis=1)

        if plot:
            fig = px.line(
                res.set_index("Дата"),
                title=title,
                labels={"value": f"Остаток портфеля, {cur}"},
            )
            fig.show()

        return res

    def forecast_balance_time_horizon(self, test, cur, start_date, months=[1, 3, 12]):
        res = []

        dates = test.open_month.unique()

        dates = np.sort(dates[dates >= start_date])

        max_date = pd.to_datetime(dates[-1]) + pd.offsets.MonthEnd(1)

        for date in dates:
            dat = self.forecast_balance_long(test, max(months), date, plot=False)

            dttm = pd.to_datetime(date) + pd.offsets.MonthEnd(1)

            port = test[test.open_month < date][test.report_month == date][
                "total_generation_cleared"
            ].sum()

            for m in months:
                if dttm + pd.offsets.MonthEnd(m) <= max_date:
                    res.append([dttm, m, port, *list(dat.iloc[m - 1, 1:])])

                else:
                    res.append(
                        [dttm, m, port, *[np.nan for _ in range(dat.shape[1] - 1)]]
                    )

        res_df = pd.DataFrame(
            res,
            columns=[
                "Дата",
                "Срок",
                "Портфель",
                "Прогноз модели",
                "Прогноз по контрактному погашению",
                "Фактические значения",
                "Прогноз досрочных погашений",
                "Досрочные погашения",
            ],
        )

        res_df[f"Моментальная ошибка модели, {cur}"] = (
            res_df["Досрочные погашения"] - res_df["Прогноз досрочных погашений"]
        )

        res_df[f"Моментальная ошибка без модели, {cur}"] = res_df["Досрочные погашения"]

        res_df["Моментальная ошибка модели, % от портфеля"] = (
            res_df[f"Моментальная ошибка модели, {cur}"] / res_df["Портфель"] * 100
        )

        res_df["Моментальная ошибка без модели, % от портфеля"] = (
            res_df[f"Моментальная ошибка без модели, {cur}"] / res_df["Портфель"] * 100
        )

        res_df[f"Накопленная ошибка модели, {cur}"] = (
            res_df["Фактические значения"] - res_df["Прогноз модели"]
        )

        res_df[f"Накопленная ошибка без модели, {cur}"] = (
            res_df["Фактические значения"] - res_df["Прогноз по контрактному погашению"]
        )

        res_df["Накопленная ошибка модели, % от портфеля"] = (
            res_df[f"Накопленная ошибка модели, {cur}"] / res_df["Портфель"] * 100
        )

        res_df["Накопленная ошибка без модели, % от портфеля"] = (
            res_df[f"Накопленная ошибка без модели, {cur}"] / res_df["Портфель"] * 100
        )

        metrics = [
            f"Моментальная ошибка модели, {cur}",
            f"Моментальная ошибка без модели, {cur}",
            "Моментальная ошибка модели, % от портфеля",
            "Моментальная ошибка без модели, % от портфеля",
            f"Накопленная ошибка модели, {cur}",
            f"Накопленная ошибка без модели, {cur}",
            "Накопленная ошибка модели, % от портфеля",
            "Накопленная ошибка без модели, % от портфеля",
        ]

        for metric in metrics:
            res_df[f"{metric} abs"] = np.abs(res_df[metric])

        metrics_abs = [metric + " abs" for metric in metrics]

        print(res_df.groupby("Срок")[metrics_abs].mean())

        res_df = res_df.sort_values(["Дата", "Срок"])

        for type_ in ("Накопленная ", "Моментальная "):
            for perc in (", % от портфеля", f", {cur}"):
                fig = px.line(
                    res_df,
                    x="Дата",
                    y=[
                        f"{type_}ошибка модели{perc}",
                        f"{type_}ошибка без модели{perc}",
                    ],
                    title=f"{type_}ошибка модели{perc}",
                    facet_row="Срок",
                    labels={"value": f"{perc[2:]}"},
                )
                fig.show()

    def plot(self, test, title="суммарный прогноз поколений"):
        forecast_cols = [x for x in test.columns if "total_generation_cl_predict" in x]
        plot_cols = ["total_generation_cleared"] + forecast_cols

        df_plot = (
            test[["report_month"] + plot_cols]
            .groupby("report_month")
            .agg({x: "sum" for x in plot_cols})
        )

        fig = px.line(df_plot.astype(np.float64))

        fig.update_layout(title=title)

        fig.show()

    def max_rates(self, dataframe, periods_list):
        rates_list = [f"report_weight_open_rate_{i}m" for i in periods_list]

        if len(rates_list) > 1:
            return dataframe[rates_list].max(axis=1)
        else:
            return dataframe[rates_list[0]]

    def feature_generation(self, data):
        # formerly known as train_test_split

        dat = data.copy()

        # for col in dat.columns:
        #     try:
        #         dat[col] = dat[col].astype(np.float64)
        #     except:
        #         pass

        dat["long_generations_flg"] = np.int64(dat["bucketed_period"] > 6)

        dat["report_weight_open_rate_1m"] = (
            dat["report_weight_open_rate_1m"]
            .fillna(dat["report_weight_open_rate_3m"])
            .astype(float)
        )

        dat["weight_rate_12m&weight_rate_3m"] = dat[
            "report_weight_open_rate_12m"
        ].astype(float) - dat["report_weight_open_rate_3m"].astype(float)
        dat["weight_rate_24m&weight_rate_3m"] = dat[
            "report_weight_open_rate_24m"
        ].astype(float) - dat["report_weight_open_rate_3m"].astype(float)

        for col in dat.columns:
            if "weight_rate_1" in col or "weight_rate_2" in col:
                dat[col] = dat[col] * dat["long_generations_flg"]
                dat[f"{col}_lag1"] = dat.groupby("gen_name")[col].shift()

        dat["spread_weight_rate_&_report_wo_period_perc_90"] = dat[
            "weight_rate"
        ].astype(float) - dat["report_wo_period_perc_90"].astype(float)
        dat["spread_weight_rate_&_report_perc_90"] = dat["weight_rate"].astype(
            float
        ) - dat["report_perc_90"].astype(float)

        dat["spread_weight_rate_&_report_perc_90_mod"] = np.where(
            dat["weight_rate_12m&weight_rate_3m"] < 0,
            dat["spread_weight_rate_&_report_wo_period_perc_90"],
            dat["spread_weight_rate_&_report_perc_90"],
        )

        dat["days_left"] = dat["days_plan"].astype(float) - dat["days_passed"].astype(
            float
        )

        dat["months_passed"] = dat["row_count"].astype(float) - 1
        dat["months_left"] = (
            dat["bucketed_period"].astype(float)
            + 1
            - dat["months_passed"].astype(float)
        )

        periods_list = [24, 12, 6, 3, 1]
        periods_list.sort(reverse=True)

        # килер фича
        dat["spread_weight_rate_&_weight_open_rate"] = dat["weight_rate"].astype(
            float
        ) - self.max_rates(dat, periods_list)
        dat["spread_weight_rate_&_weight_open_rate_div"] = dat[
            "spread_weight_rate_&_weight_open_rate"
        ].astype(float) / dat["weight_rate"].astype(float)

        dat["SER_dinamic_cl_lag1"] = (
            dat.groupby("gen_name")["SER_dinamic_cl"].shift().astype(float)
        )
        dat["spread_weight_rates_wo_period_opt"] = dat["weight_rate"].astype(
            float
        ) - dat["report_wo_period_wo_opt_weight_open_rate"].astype(float)
        dat["spread_weight_rates_wo_period"] = dat["weight_rate"].astype(float) - dat[
            "report_wo_period_weight_open_rate"
        ].astype(float)

        # для безопц немного другая фича:
        # dat['incentive'] = (
        #     dat['weight_rate'] * (dat['bucketed_period'] + 1) - dat['months_left'] * self.max_rates(dat, periods_list)
        #  ) / 12

        # Для вкладов с опцией снятия
        dat.loc[dat["optional_flg"].isin([1, 3]), "incentive"] = dat.loc[
            dat["optional_flg"].isin([1, 3]), "spread_weight_rates_wo_period"
        ]

        # Для вкладов без опции снятия
        dat.loc[dat["optional_flg"].isin([0, 2]), "incentive"] = dat.loc[
            dat["optional_flg"].isin([0, 2]), "spread_weight_rate_&_weight_open_rate"
        ]

        dat["incentive_lag1"] = dat.groupby("gen_name")["incentive"].shift()
        dat["volume_share"] = (
            dat["total_generation_lag1"] / dat["init_total_generation"]
        )
        dat["volume_share_lag"] = dat.groupby("gen_name")["volume_share"].shift()

        dat["outlier_flg"] = 0
        for col in dat.columns:
            if "spread" in col:
                dat[f"{col}_lag1"] = dat.groupby("gen_name")[col].shift()
                dat["outlier_flg"] = np.where(
                    (dat[col] <= dat[col].quantile(0.01))
                    | (dat[col] >= dat[col].quantile(0.99)),
                    1,
                    dat["outlier_flg"],
                )

        # Если вклады с опцией пополнения - выбросом считаем увеличение более чем в 10 раз
        # Если вклады без опции пополнения - выбросы при обычном увеличении (не должны увеличиваться при условии вычищенных процентов)
        dat["outlier_flg"] = (dat["optional_flg"].isin([0, 1])) * np.where(
            dat["SER_dinamic_cl"] > 0, 1, dat["outlier_flg"]
        ) + (dat["optional_flg"].isin([2, 3])) * np.where(
            dat["SER_dinamic_cl"] > 10, 1, dat["outlier_flg"]
        )

        # data_train = dat[(dat.report_month >= '2014-01') & (dat.report_month < split_date)]
        # data_test = dat[(dat.report_month >= split_date) & (dat.report_month < dat.open_month.max())]

        # TODO: добавить фильтры из early_redemption_v1
        ## Подумать наж этой штукой (если будет использовать это в предикте - будет плохо)
        dat = dat[dat.count_agreements > 10]

        outlier_flag_new = dat.groupby("gen_name")["outlier_flg"].transform("max")
        dat = dat[~(outlier_flag_new == 1)].reset_index(drop=True)

        return dat

    @staticmethod
    def PTR_input_transform(split_columns, split_labels, split_tresholds):
        if len(split_columns) == 0:
            return [[], [], []]

        else:
            split_df = pd.DataFrame([split_columns, split_labels, split_tresholds]).T

            split_df.columns = ["cols", "labels", "tresholds"]

            split_df = (
                split_df.groupby(["cols", "labels"], sort=False)["tresholds"]
                .unique()
                .reset_index()
            )
            # split_df = split_df.groupby(['cols', 'labels'])['tresholds'].unique().reset_index()

            split_df["tresholds"] = split_df["tresholds"].apply(lambda x: x.tolist())

            output = [split_df[x].values.tolist() for x in split_df.columns]

            return output

    def eval_for_nm(
        self,
        data_train,
        features,
        cols_to_split_on,
        labels,
        n_iter,
        p_step,
        p_low,
        p_high,
        target,
        init_cols=[],
        init_labels=[],
        init_tresh=[],
        add_constant=True,
        scaler=False,
        entity_effects=True,
        lasso_alpha=None,
        rank_check_in_iters=True,
    ):
        pass

    def LMT(
        self,
        data_train,
        features,
        cols_to_split_on,
        labels,
        n_iter,
        p_step,
        p_low,
        p_high,
        target,
        weights,
        init_cols=[],
        init_labels=[],
        init_tresh=[],
        add_constant=True,
        scaler=False,
        entity_effects=False,
        lasso_alpha=None,
        rank_check_in_iters=True,
    ):
        # self.LMT(data_train, self.features, self.cols,
        # self.labels, n_iter=self.n_iter, target=self.target)
        init_PTR_input = self.PTR_input_transform(init_cols, init_labels, init_tresh)

        # print(init_PTR_input)
        # print(features)
        # print(data_train.columns)

        init_train_ptr, columns = early_redemption_ptr_transform(
            data_train,
            init_PTR_input[0],
            init_PTR_input[1],
            init_PTR_input[2],
            exog=features,
        )

        reg = self.lin_reg(
            train=init_train_ptr,
            # test=init_train_ptr,
            features=columns,
            target=target,
            weights=weights,
            scaler=scaler,
            add_constant=add_constant,
            lasso_alpha=lasso_alpha,
            print_summary=False,
            entity_effects=entity_effects,
            rank_check=rank_check_in_iters,
        )

        init_score = reg.rsquared
        best_score = init_score
        print(f"initial score = {init_score}")

        col_results = []

        percentiles = np.arange(p_low, p_high + p_step, p_step)
        perc_df = data_train.quantile(percentiles / 100)

        for i, col in enumerate(cols_to_split_on):
            # dat = data_train[col].sort_values()

            # percentiles = np.arange(p_low, p_high+p_step, p_step)

            best_col_res = 0
            print(f"\t split on {labels[i]}")

            perc_last = None

            # for perc in percentiles:
            for perc in perc_df[col].values:
                if perc == perc_last:
                    perc_last = perc
                    continue

                # t = dat.iloc[int(perc * len(dat) / 100)]

                split_cols = init_cols + [col]

                # split_tresh = init_tresh + [t]
                split_tresh = init_tresh + [perc]

                split_lab = init_labels + [labels[i]]

                PTR_input = self.PTR_input_transform(split_cols, split_lab, split_tresh)

                dat_train_ptr, columns = early_redemption_ptr_transform(
                    data_train, PTR_input[0], PTR_input[1], PTR_input[2], exog=features
                )
                try:
                    reg = self.lin_reg(
                        train=dat_train_ptr,
                        # test=init_train_ptr,
                        features=columns,
                        target=target,
                        weights=weights,
                        scaler=scaler,
                        add_constant=add_constant,
                        lasso_alpha=lasso_alpha,
                        print_summary=False,
                        entity_effects=entity_effects,
                        rank_check=rank_check_in_iters,
                    )
                except ValueError as e:
                    print("failed to fit (this is ok)! Reason:")
                    print(e)

                    # with open('dat_train.pickle', 'wb') as f:
                    #     pickle.dump([dat_train_ptr, columns, target, weights, perc], f)

                    # raise e
                    continue

                new_score = reg.rsquared
                print(f"\t\t split_tresh={split_tresh[-1]}, new_score={new_score}")

                if new_score > best_col_res:
                    best_col_res = new_score
                    best_col_tresh = split_tresh

                perc_last = perc

            # if best_col_res <= best_score:
            if best_col_res <= init_score:
                print("\t\t no good splits")
                continue

            best_score = best_col_res
            print(
                f"\t\t best tresh = {best_col_tresh[-1]}, best score = {best_col_res}"
            )
            col_results.append([split_cols, split_lab, best_col_tresh, best_col_res])

        col_results.sort(key=lambda x: x[-1], reverse=True)

        if len(col_results) == 0:
            print("No good splits were found. Previous iteration params are returned")
            return init_cols, init_labels, init_tresh

        print(f"best split: {col_results[0]}")
        # было 0.03
        if col_results[0][-1] - init_score > 0.01 and n_iter > 1:
            return self.LMT(
                data_train,
                features,
                cols_to_split_on,
                labels,
                n_iter - 1,
                p_step,
                p_low,
                p_high,
                target,
                weights,
                col_results[0][0],
                col_results[0][1],
                col_results[0][2],
                add_constant=add_constant,
                scaler=scaler,
                entity_effects=entity_effects,
                lasso_alpha=lasso_alpha,
                rank_check_in_iters=rank_check_in_iters,
            )

            # LMT(self, data_train, features, cols_to_split_on, labels,
            # n_iter, p_step, p_low, p_high, target, weights,
            # init_cols=[], init_labels=[], init_tresh=[],
            # add_constant=True, scaler=False,
            # entity_effects=True, lasso_alpha=None,
            # rank_check_in_iters=True):
        elif n_iter <= 1:
            print("Max iterations reached. Stop iteration process")
            return col_results[0][0], col_results[0][1], col_results[0][2]
        else:
            print(
                "Score imporvement is below 0.01. Previous iteration params are returned"
            )
            return col_results[0][0], col_results[0][1], col_results[0][2]

    def fit(
        self,
        train_data,
        n_iter,
        p_step,
        p_low,
        p_high,
        add_constant,
        scaler,
        entity_effects,
        lasso_alpha,
        rank_check_in_iters,
    ):
        if train_data.shape[0] < 1:
            model_info = [None, self.target, self.leaf_reg_features, [], [], []]
            return model_info

        # if add_constant:
        #     train_data.loc[:, 'const'] = 1
        #     self.leaf_reg_features += ['const']

        # spark = run_spark_session('Build dataframe for dep_er_no_opt training')
        # data = spark.sql(
        # f'''
        # select * from dadm_alm_sbx.early_close_prod_VP_30n_vip_share_with_droped
        # where CUR = "{self.cur}" and optional_flg = 0 and close_month > report_month and count_agreements > 40 and drop_flg = 0
        # ''')
        # data = self.data_transform(data, self.cur)
        # spark.stop()

        # data_train, data_test = self.train_test_split(train_data, self.date)
        data_train = self.feature_generation(train_data)

        # LMT(self, data_train, features, cols_to_split_on, labels,
        #    n_iter, p_step, p_low=2, p_high=98, target='SER_dinamic_cl', init_cols=[], init_labels=[], init_tresh=[])

        if n_iter:
            self.n_iter = n_iter

        m_cols, m_labels, m_tresh = self.LMT(
            data_train,
            self.leaf_reg_features,
            self.splitting_cols,
            self.splitting_cols_labels,
            n_iter=n_iter,
            p_step=p_step,
            p_low=p_low,
            p_high=p_high,
            target=self.target,
            weights=self.weight_col,
            add_constant=add_constant,
            scaler=scaler,
            entity_effects=entity_effects,
            lasso_alpha=lasso_alpha,
            rank_check_in_iters=rank_check_in_iters,
        )

        data_train_ptr, columns = early_redemption_ptr_transform(
            data_train,
            m_cols,
            m_labels,
            [[x] for x in m_tresh],
            exog=self.leaf_reg_features,
        )
        # data_test_ptr, columns = early_redemption_ptr_transform(data_test, m_cols, m_labels, [[x] for x in m_tresh], exog=self.features)

        # if predict:
        #     reg, [data_train_res, data_test_res] = self.lin_reg(data_train_ptr, data_test_ptr, columns,
        #                                                self.target, add_constant=False, lasso_alpha=self.alpha, predict=predict)
        reg = self.lin_reg(
            train=data_train_ptr,
            features=columns,
            target=self.target,
            weights=self.weight_col,
            add_constant=add_constant,
            lasso_alpha=lasso_alpha,
            print_summary=True,
            entity_effects=entity_effects,
            scaler=False,
            rank_check=True,
        )

        model_info = [
            reg,
            self.target,
            self.leaf_reg_features,
            m_cols,
            m_labels,
            m_tresh,
        ]

        return model_info
