from enum import Enum
from pickle import load
from datetime import datetime
from calendar import monthrange
from pandas import DataFrame, melt
from abc import ABC, abstractmethod
from pyspark.sql import SparkSession
from dataclasses import dataclass, field
from pandas.tseries.offsets import MonthEnd
from typing import Dict, Any, Tuple, List, Type
from dateutil.relativedelta import relativedelta

_REPORT_DT_COLUMN = "report_dt"


class Products(Enum):
    pass


@dataclass(frozen=True)
class TrainingPeriod:
    start_dt: datetime
    end_dt: datetime

    def __str__(self) -> str:
        dateformat = "%Y%m"
        return (
            f'{self.start_dt.strftime(dateformat) if self.start_dt else ""}'
            f'{"_" + self.end_dt.strftime(dateformat) if self.end_dt else ""}'
        )


@dataclass(frozen=True)
class ModelInfo:
    prefix: str
    training_period: TrainingPeriod = None

    def __str__(self) -> str:
        if self.training_period:
            return f"{self.prefix}_{str(self.training_period)}"

        return f"{self.prefix}_"

    @property
    def model_key(self):
        return ModelInfo(self.prefix)

    @staticmethod
    def from_str(text: str, separator: str = "_", dateformat="%Y%m"):
        info_: ModelInfo = ModelInfo(text)
        try:
            parts: List[str] = text.split(separator)
            training_period = None
            from_dt: datetime = datetime.strptime(parts[-2], dateformat)
            to_dt: datetime = datetime.strptime(parts[-1], dateformat)
            training_period = TrainingPeriod(from_dt, to_dt)
            prefix_: str = separator.join(parts[:-2])
            info_: ModelInfo = ModelInfo(prefix_, training_period)
        except Exception as e:
            pass

        return info_


class MLException(Exception):
    """Ошибка во время работы модели"""

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass


class Scenario:
    """
    Сценарий для прогнозов

    Parameters
    ----------
    portfolio_dt: datetime
        Дата портфолио
    forecast_horizon: int
        Горизонт прогноза в месяцах
    scenario_data: DataFrame
        Датафрейм с данными сценария
    """

    def __init__(
        self, portfolio_dt: datetime, horizon: int, scenario_data: DataFrame
    ) -> None:
        self._portfolio_dt: datetime = portfolio_dt
        self._horizon = horizon
        self._scenario_data: DataFrame = scenario_data

    @property
    def forecast_dates(self) -> List[datetime]:
        dates_: List[datetime] = [
            (self.portfolio_dt + relativedelta(months=m))
            for m in range(1, self._horizon + 1)
        ]
        return [dt.replace(day=monthrange(dt.year, dt.month)[1]) for dt in dates_]

    @property
    def scenario_data(self) -> DataFrame:
        return self._scenario_data

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def portfolio_dt(self) -> datetime:
        return self._portfolio_dt

    def subscenario(self, forecast_date: datetime):
        return Scenario(
            portfolio_dt=(forecast_date + MonthEnd(-1)).to_pydatetime(),
            horizon=1,
            scenario_data=self._scenario_data.loc[[forecast_date], :],
        )

    def unpivot_data(self, column_prefix: str, col_name: str = "name") -> DataFrame:
        col_names = ["report_dt"] + [
            col_name
            for col_name in self.scenario_data_.columns
            if col_name.startswith(column_prefix)
        ]
        return melt(
            self.scenario_data_[col_names], id_vars=["report_dt"], value_name=col_name
        )


@dataclass
class ForecastContext:
    """
    Контекст прогноза

    Parameters
    ----------
    portfolio_dt: datetime = None
        Дата портфолио
    forecast_horizon: int = 0
        Горизонт прогноза в месяцах
    scenario: Scenario = None
        Обьект сценария. Содержит в себе датафрейм scenario_data с данными сценария
    model_data: Dict[str, Any] = None
        Словарь с предсказаниями от других моделей.
        Заполняется по ходу исполнения вычислений в калькуляторе
    """

    portfolio_dt: datetime = None  # TODO: `scenario` contains `portfolio_dt`
    forecast_horizon: int = 0  # TODO: `scenario` contains `forecast_horizon`
    scenario: Scenario = None
    model_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def forecast_dates(self) -> List[datetime]:
        return self.scenario.forecast_dates

    def subcontext(self, forecast_date):
        return ForecastContext(
            portfolio_dt=(forecast_date + MonthEnd(-1)).to_pydatetime(),
            forecast_horizon=1,
            scenario=self.scenario.subscenario(forecast_date),
            model_data=self.model_data,
        )


class BaseModel(ABC):
    """
    Абстрактный класс адаптера моделей

    Для реализации интерфейса, необходимо реализовать метод `predict`
    """

    def __init__(self, model_info_: ModelInfo, filepath_or_buffer) -> None:
        self._model_info: ModelInfo = model_info_
        self._model_meta = self._unpickle_file_or_buffer(filepath_or_buffer)

    @abstractmethod
    def predict(
        self, forecast_context: ForecastContext, portfolio: DataFrame = None, **params
    ) -> Any:
        """
        Возвращает предсказание десериализованной модели в произвольном формате

        Parameters
        ----------
        forecast_context : ForecastContext
            Контекст прогноза. Содержит в себе сценарий для прогноза, а также
            (опционально) прогнозы других моделей. См. описание класса ForecastContext
        portfolio : DataFrame, default = None
            Датафрейм с дополнительными данными из дата провайдера
        params
            Произвольные параметры адаптера

        Returns
        -------
        prediction : Any
            Предсказание модели в произвольной форме
        """
        pass

    def _unpickle_file_or_buffer(self, filepath_or_buffer) -> Any:
        """
        Читает модель из файла или файла подобного объекта, возвращает объект модели

        Parameters
        ----------
        filepath_or_buffer : файл или подобный файлу объект
            путь к файлу или объект у которого есть метод read, например StringIO или BytesIO
        Returns
        -------
        obj : Any
            Объект модели
        """
        if hasattr(filepath_or_buffer, "read"):
            return load(filepath_or_buffer)

        with open(filepath_or_buffer, "rb") as in_file:
            return load(in_file)

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info

    @property
    def model(self) -> Any:
        return self._model_meta


class ModelTrainer(ABC):
    """
    Абстрактный класс тренера моделей

    Для реализации интерфейса, необходимо реализовать методы `get_trained_model` и `save_trained_model`
    """

    @abstractmethod
    def get_trained_model(
        self,
        spark: SparkSession,
        end_date: datetime,
        start_date: datetime = None,
        hyperparams: Dict[str, Any] = None,
    ) -> Any:
        """
        Обучает и возвращает модель с заданными параметрами

        Parameters
        ----------
        spark : spark-сессия
            Сессия для загрузки данных
        end_date : datetime
            Последний день периода, на котором обучается модель
        start_date : datetime, default=None
            Первый день периода, на котором обучается модель. Реализация метода должна обрабатывать значение по умолчанию None.
        hyperparams : Dict[str, Any], default=None
            Гиперпараметры обучения модели. В реализации могут отсутствовать, либо реализация должна обрабатывать
            значение по умолчанию None (например загружать дефолтные гиперпараметры из глобальных переменных
             или других источников).

        Returns
        -------
        модель : Any
            Произвольный класс обученной модели
        """
        pass

    @abstractmethod
    def save_trained_model(
        self,
        spark: SparkSession,  # TODO: already have in __init__
        saving_path: str,
        end_date: datetime,  # TODO: already have in __init__
        start_date: datetime = None,  # TODO: already have in __init__
        overwrite: bool = True,
        hyperparams: Dict[str, Any] = None,  # TODO: already have in __init__
    ) -> str:
        """
        Обучает и сохраняет модель с заданными параметрами. При этом должна
        соблюдаться конвенция наименований сохраненных файлов

        Parameters
        ----------
        spark : spark-сессия
            Сессия для загрузки данных
        saving_path: str
            Папка, куда сохранится модель
        end_date : datetime
            Последний день периода, на котором обучается модель
        start_date : datetime, default=None
            Первый день периода, на котором обучается модель. Реализация метода
            должна обрабатывать значение по умолчанию None.
        overwrite : bool, default=True
            Флаг перезаписи модели. Если True, то метод должен перезаписывать
            существующий файл модели. Если False - то пропускать обучение, если
            файл уже есть, и сразу возвращать название файла.
        hyperparams : Dict[str, Any], default=None
            Гиперпараметры обучения модели. В реализации могут отсутствовать,
            либо реализация должна обрабатывать значение по умолчанию None
            (например загружать дефолтные гиперпараметры из глобальных переменных
            или других источников).

        Returns
        -------
        file_name : str
            Возвращает имя сохраненной модели в случае успеха обучения и
            сохранения модели, иначе - None
        """
        pass


class DataLoader(ABC):
    """
    Абстрактный класс загрузчика данных.

    Для реализации интерфейса, необходимо реализовать методы get_maximum_train_range, get_training_data,
    get_prediction_data и get_ground_truth
    """

    @abstractmethod
    def get_maximum_train_range(self, spark: SparkSession) -> Tuple[datetime, datetime]:
        """
        Возвращает максимальный доступный период обучения модели.
        Рекомендуется определять минимальную дату начала обучения из соображений валидности модели на старых периодах,
        а не из доступности данных (но проверка доступности все равно нужна).
        Максимальную дату конца обучения, наоборот, следует определять из доступности самых последних данных.

        Parameters
        ----------
        spark : spark-сессия
            Сессия для загрузки данных

        Returns
        -------
        maximum_train_range : Tuple[datetime, datetime]
            Кортеж вида (start_date, end_date).
        """
        pass

    @abstractmethod
    def get_training_data(
        self,
        spark: SparkSession,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, DataFrame]:
        """
        Возвращает данные для обучения модели: фичи и таргет

        Parameters
        ----------
        spark : spark-сессия
            Сессия для загрузки данных
        start_date : datetime
            Первый день периода загружаемых данных
        end_date : datetime
            Последний день периода загружаемых данных
        params : Dict[str, Any], default=None
            Параметры загрузки данных. В реализации могут отсутствовать, либо реализация должна обрабатывать
            значение по умолчанию None (например загружать дефолтные параметры из глобальных переменных
             или других источников).

        Returns
        -------
        training_data : Dict[str, DataFrame]
            Словарь с данными для обучения модели произвольного вида (например {'target': DataFrame, 'features1': DataFrame, 'features2': DataFrame, ...})
        """
        pass

    @abstractmethod
    def get_prediction_data(
        self,
        spark: SparkSession,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, DataFrame]:
        """
        Возвращает реализовавшийся сценарий фичей для оценки точности модели на истории (бектеста).

        Этот метод должен возвращать только те фичи, которые невозможно посчитать для будущего!
        Если фича моделируется другой моделью, то она может считаться доступной в ForecastContext.model_data
        Если фича может быть посчитана из текущего состояния (например портфолио на дату, плановые оттоки и д.р.),
        то она должна быть загружена через метод get_portfolio, либо через отдельную модель.

        Крайне рекомендуется загружать здесь фичи без предобработки,
        чтобы аналогичный по атрибутному составу сценарий можно было бы заполнить
        руками при необходимости. Предобработку следует делать либо в предикте модели, либо в ModelAdapter

        Набор фичей может не совпадать с таковым на обучении (например, если обучались для таргета в виде % прироста,
        а предсказывать хочется изначальный ряд - тогда одна из фичей должна содержать последнее известное значение ряда).


        Также необходимо соблюдать формат возвращаемого словаря (см. ниже).

        Parameters
        ----------
        spark : spark-сессия
            Сессия для загрузки данных
        start_date : datetime
            Первый день периода загружаемых данных
        end_date : datetime
            Последний день периода загружаемых данных
        params : Dict[str, Any], default=None
            Параметры загрузки данных. В реализации могут отсутствовать, либо реализация должна обрабатывать
            значение по умолчанию None (например загружать дефолтные параметры из глобальных переменных
             или других источников).

        Returns
        -------
        prediction_data : Dict[str, DataFrame]
            Словарь с несколькими ключами. Ключ 'features' используется для датафрейма с данными в формате:
                report_dt - даты концов месяцев, содержащиеся в периоде от start_date до end_date
                feature_name1 - название первой фичи, должно совпадать с названием
                колонки в Scenario.scenario_data, которую будет использовать ModelAdapter при предсказании
                feature_name2, feature_name3, и т.д. - аналогично feature_name1
            Могут использоваться дополнительные ключи если это необходимо.
        """
        pass

    @abstractmethod
    def get_ground_truth(
        self,
        spark: SparkSession,
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, DataFrame]:
        """
        Возвращает реализовавшийся сценарий таргета для оценки точности модели на истории (бектеста).

        Может не совпадать с таргетом в обучении (например, если обучались для таргета в виде % прироста,
        а предсказывается изначальный ряд).

        Также необходимо соблюдать формат возвращаемого словаря (см. ниже).

        Parameters
        ----------
        spark : spark-сессия
            Сессия для загрузки данных
        start_date : datetime
            Первый день периода загружаемых данных
        end_date : datetime
            Последний день периода загружаемых данных
        params : Dict[str, Any], default=None
            Параметры загрузки данных. В реализации могут отсутствовать, либо реализация должна обрабатывать
            значение по умолчанию None (например загружать дефолтные параметры из глобальных переменных
             или других источников).

        Returns
        -------
        ground_truth : Dict[str, DataFrame]
            Словарь с одним ключем 'target'. По ключу - датафрейм с данными в формате:
            report_dt - даты концов месяцев, содержащиеся в периоде от start_date до end_date
            target_var1, target_var2 и т.д. - столбцы таргета.
        """
        pass

    def get_portfolio(
        self,
        spark: SparkSession,
        report_date: datetime,
        params: Dict[str, Any] = None,
    ) -> Dict[str, DataFrame]:
        """
        Возвращает портфель на отчетную дату, либо ничего.

        Метод не обязателен к имплементации в наследниках.

        Также необходимо соблюдать формат возвращаемого словаря (см. ниже).

        Parameters
        ----------
        spark : spark-сессия
            Сессия для загрузки данных
        report_date : datetime
            Отечтная дата, на которую будет загружен портфель
        params : Dict[str, Any], default=None
            Параметры загрузки данных. В реализации могут отсутствовать, либо реализация должна обрабатывать
            значение по умолчанию None (например загружать дефолтные параметры из глобальных переменных
             или других источников).

        Returns
        -------
        portfolio_dict : Dict[str, DataFrame]
            Словарь с одним или несколькими ключами, которые должны заканчиваться постфиксом "_portfolio".
            Значения - DataFrame произвольного формата.
            Этот словарь будет добавлен в ForecastContext.model_data до начала прогнозирования
        """
        return None


@dataclass(frozen=True)
class ModelMetaInfo:
    model_name: str
    model_trainer: Type[ModelTrainer]
    data_loader: Type[DataLoader]
    adapter: Type[BaseModel]
    segment: str = None
    replenishable_flg: int = None
    subtraction_flg: int = None


class ModelContainer:
    def __init__(
        self,
        models: Tuple[ModelMetaInfo] = None,
        model_containers: Tuple["ModelContainer"] = None,
    ):
        if models is not None:
            self.models: List[ModelMetaInfo] = list(models)
        else:
            self.models = []

        if model_containers is not None:
            for model_container in model_containers:
                for model_ in model_container.models:
                    self.models.append(model_)

    @property
    def model_names(self) -> List[str]:
        return [model.model_name for model in self.models]

    @property
    def models_dict(self) -> Dict[str, ModelMetaInfo]:
        return {model.model_name: model for model in self.models}

    @property
    def trainers(self) -> Dict[str, ModelTrainer]:
        return {model.model_name: model.model_trainer for model in self.models}

    @property
    def dataloaders(self) -> Dict[str, DataLoader]:
        return {model.model_name: model.data_loader for model in self.models}

    @property
    def adapter_types(self) -> Dict[str, DataLoader]:
        return {model.model_name: model.adapter for model in self.models}

    def get_model_by_name(self, model_name: str):
        return self.models_dict.get(model_name, None)

    def get_model_by_conditions(self, **conditions):
        if conditions == {}:
            raise KeyError("Conditions are not defined")

        model_counter = 0
        for model in self.models:
            if all([getattr(model, cond) == conditions[cond] for cond in conditions]):
                if model_counter < 1:
                    match = model
                    model_counter += 1
                else:
                    raise KeyError(
                        f"More than one model has conditions like {conditions}"
                    )
        if model_counter > 0:
            return match
        else:
            return None


@dataclass(frozen=True)
class PackageMetaInfo:
    models: Tuple[ModelMetaInfo]
    author: str = "upfm"
    email: str = "upfm@vtb.ru"

    def list_models(self):
        print("index \t | \t model_name")
        print("---------------------------------")
        for i, m in enumerate(self.models):
            print(f"{i} \t | \t {m.model_name}")
