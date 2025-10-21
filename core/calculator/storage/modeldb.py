import logging
import pandas as pd
from typing import Dict, List
from datetime import datetime
from collections import defaultdict
from os.path import splitext, basename
from sqlalchemy import (
    Column,
    Integer,
    Float,
    ForeignKey,
    String,
    DateTime,
    LargeBinary,
    create_engine,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref, sessionmaker

from core.calculator.core.settings import Settings


logging.config.dictConfig(Settings.LOGGING_CONFIG)
logger = logging.getLogger("core")


from core.upfm.commons import ModelInfo


Base = declarative_base()


class ModelInfoEntity(Base):
    __tablename__ = "model_info"

    model_info_id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    mart = Column(String)
    saved_at = Column(DateTime)
    git_url = Column(String)

    trained_models = relationship("TrainedModelEntity", backref=backref("model_info"))
    portfolios = relationship("PortfolioEntity", backref=backref("model_info"))
    prediction_data = relationship("PredictionEntity", backref=backref("model_info"))
    ground_truth = relationship("GroundTruthEntity", backref=backref("model_info"))

    def __repr__(self) -> str:
        return f"id = {self.model_info_id} {self.name}"

    def __str__(self) -> str:
        return f'{self.name} {" ".join([str(tm) for tm in self.trained_models])}'


class TrainedModelEntity(Base):
    __tablename__ = "trained_model"

    trained_model_id = Column(Integer, primary_key=True)
    saved_at = Column(DateTime)
    model_info_id = Column(
        Integer,
        ForeignKey("model_info.model_info_id", ondelete="CASCADE"),
    )
    train_start_dt = Column(DateTime)
    train_end_dt = Column(DateTime)
    file_name = Column(String, unique=True)
    model_data = Column(LargeBinary)

    # TODO: make this column deffered
    UniqueConstraint(
        "model_info_id",
        "train_start_dt",
        "train_end_dt",
        name="unqiue_traned_model",
    )

    def to_model_info(self) -> ModelInfo:
        return ModelInfo.from_str(self.file_name)

    def __str__(self) -> str:
        return f"{str(self.to_model_info())}"

    def __repr__(self) -> str:
        return f"id = {self.trained_model_id} {str(self.to_model_info())}"


class PredictionEntity(Base):
    __tablename__ = "prediction"

    prediction_id = Column(Integer, primary_key=True)
    saved_at = Column(DateTime)
    model_info_id = Column(
        Integer,
        ForeignKey("model_info.model_info_id", ondelete="CASCADE"),
    )
    start_dt = Column(DateTime)
    end_dt = Column(DateTime)

    prediction_data = relationship(
        "PredictionDataEntity", backref=backref("prediction")
    )

    UniqueConstraint(
        "model_info_id",
        "start_dt",
        "end_dt",
        name="unqiue_prediction",
    )


class PredictionDataEntity(Base):
    __tablename__ = "prediction_data"

    prediction_data_id = Column(Integer, primary_key=True)
    saved_at = Column(DateTime)
    prediction_id = Column(
        Integer,
        ForeignKey("prediction.prediction_id", ondelete="CASCADE"),
    )
    tag = Column(String)
    protfolio_data = Column(LargeBinary)

    UniqueConstraint(
        "prediction_id",
        "tag",
        name="unqiue_portfolio_data",
    )


class PortfolioEntity(Base):
    __tablename__ = "portfolio"

    portfolio_id = Column(Integer, primary_key=True)
    saved_at = Column(DateTime)
    model_info_id = Column(
        Integer,
        ForeignKey("model_info.model_info_id", ondelete="CASCADE"),
    )
    portfolio_dt = Column(DateTime)

    portfolio_data = relationship("PortfolioDataEntity", backref=backref("portfolio"))

    UniqueConstraint(
        "model_info_id",
        "portfolio_dt",
        name="unqiue_portfolio",
    )


class PortfolioDataEntity(Base):
    __tablename__ = "portfolio_data"

    portfolio_data_id = Column(Integer, primary_key=True)
    saved_at = Column(DateTime)
    portfolio_id = Column(
        Integer,
        ForeignKey("portfolio.portfolio_id", ondelete="CASCADE"),
    )
    tag = Column(String)
    protfolio_data = Column(LargeBinary)

    UniqueConstraint(
        "portfolio_id",
        "tag",
        name="unqiue_portfolio_data",
    )


class GroundTruthEntity(Base):
    __tablename__ = "ground_truth"

    ground_truth_id = Column(Integer, primary_key=True)
    saved_at = Column(DateTime)
    model_info_id = Column(
        Integer, ForeignKey("model_info.model_info_id", ondelete="CASCADE")
    )
    start_dt = Column(DateTime)
    end_dt = Column(DateTime)

    ground_truth_data = relationship(
        "GroundTruthDataEntity", backref=backref("ground_truth")
    )

    UniqueConstraint("model_info_id", "start_dt", "end_dt", name="unqiue_ground_truth")


class GroundTruthDataEntity(Base):
    __tablename__ = "ground_truth_data"

    ground_truth_data_id = Column(Integer, primary_key=True)
    saved_at = Column(DateTime)
    ground_truth_id = Column(
        Integer, ForeignKey("ground_truth.ground_truth_id", ondelete="CASCADE")
    )
    tag = Column(String)
    truth_data = Column(LargeBinary)

    UniqueConstraint("ground_truth_id", "tag", name="unqiue_ground_truth_data")


class BackTestInfoEntity(Base):
    __tablename__ = "backtest_info"

    backtest_info_id = Column(Integer, primary_key=True)
    saved_at = Column(DateTime)
    first_train_dt = Column(DateTime)
    steps = Column(Integer)
    horizon = Column(Integer)
    calculator = Column(String)
    calc_type = Column(String)
    tag = Column(String)

    backtest_data = relationship("BackTestDataEntity", backref=backref("backtest_info"))

    def __str__(self) -> str:
        return f'{self.calculator}_first_train={self.first_train_dt.strftime("%Y-%m")}_steps={self.steps}_h={self.horizon}'

    def __repr__(self) -> str:
        return f"id = {self.backtest_info_id} {self.__str__()}"

    @property
    def calculated_data(self):
        df: pd.DataFrame = pd.DataFrame.from_records(
            [bd.asdict for bd in self.backtest_data]
        )
        df["tag"] = self.tag
        return df


class BackTestDataEntity(Base):
    __tablename__ = "backtest_data"

    backtest_data_id = Column(Integer, primary_key=True)
    backtest_info_id = Column(
        Integer,
        ForeignKey("backtest_info.backtest_info_id", ondelete="CASCADE"),
    )
    report_dt = Column(DateTime)
    train_last_date = Column(DateTime)
    pred = Column(Float)
    truth = Column(Float)
    backtest_step = Column(Integer)
    periods_ahead = Column(Integer)
    product = Column(String)
    units = Column(String)
    segment = Column(String)
    tag = Column(String)

    @property
    def asdict(self):
        return {k: v for k, v in self.__dict__.items() if k != "_sa_instance_state"}


class ModelDB:
    def __init__(self, db_url: str):
        self._db_url: str = db_url
        self._db_engine = create_engine(db_url)
        self._session_maker = sessionmaker()
        self._session_maker.configure(bind=self._db_engine)
        self._db_session = self._session_maker()

        Base.metadata.create_all(self._db_engine)

    def find_models(self, model_filter: str = "") -> List[ModelInfoEntity]:
        rs = self._db_session.query(ModelInfoEntity)
        if model_filter:
            rs = rs.filter(ModelInfoEntity.name.like(f"%{model_filter}%")).all()
        return [c for c in rs]

    def find_model(self, model_name: str) -> ModelInfoEntity:
        entity: ModelInfoEntity = None
        rs = self._db_session.query(ModelInfoEntity).filter(
            ModelInfoEntity.name == model_name
        )
        rs_list = [c for c in rs]
        if len(rs_list) > 0:
            entity = rs_list[0]

        return entity

    def delete_model(self, model_name: str) -> bool:
        entity: ModelInfoEntity = self.find_model(model_name)
        return self._delete_entity(entity)

    def delete_trained_models(self, file_name: str) -> bool:
        entity: TrainedModelEntity = self.find_trained_model(file_name)
        return self._delete_entity(entity)

    def _delete_entity(self, entity: Base) -> bool:
        rs: bool = False
        if entity:
            try:
                self._db_session.delete(entity)
                self._db_session.commit()
                rs = True
            except:
                self._db_session.rollback()
                raise
        return rs

    def delete_backtest_by_tag(self, tag: str) -> None:
        rs = self.find_backtest(tag)
        if len(rs) > 0:
            for e in rs:
                self._delete_entity(e)

    def find_trained_model(self, file_name: str) -> TrainedModelEntity:
        entity: TrainedModelEntity = None
        rs = self._db_session.query(TrainedModelEntity).filter(
            TrainedModelEntity.file_name == file_name
        )
        rs_list = [c for c in rs]
        if len(rs_list) > 0:
            entity = rs_list[0]

        return entity

    ####

    #     def find_models(self, model_filter: str = '') -> List[ModelInfoEntity]:
    #         rs = self._db_session.query(ModelInfoEntity)
    #         if model_filter:
    #             rs = rs.filter(ModelInfoEntity.name.like(f'%{model_filter}%')).all()
    #         return [c for c in rs]

    ####

    def find_trained_model_by_dt(
        self,
        model_name: str,
        end_dt: datetime,
    ) -> TrainedModelEntity:
        entity: TrainedModelEntity = None
        x_dt = end_dt.replace(day=1)
        rs = (
            self._db_session.query(TrainedModelEntity)
            .filter(
                TrainedModelEntity.model_info.has(ModelInfoEntity.name == model_name)
            )
            .filter(TrainedModelEntity.train_end_dt == x_dt)
        )

        rs_list = [c for c in rs]
        assert len(rs_list) <= 1
        if len(rs_list) > 0:
            entity = rs_list[0]

        return entity

    def find_trained_model_by_dt1(
        self,
        end_dt: datetime,
        model_name: str = "",
    ) -> List[TrainedModelEntity]:
        x_dt = end_dt.replace(day=1)
        rs = (
            self._db_session.query(TrainedModelEntity)
            .filter(TrainedModelEntity.train_end_dt == x_dt)
            .filter(
                TrainedModelEntity.model_info.has(
                    ModelInfoEntity.name.like(f"%{model_name}%")
                )
            )
            .all()
        )

        return [c for c in rs]

    def save_model_info(
        self,
        name: str,
        mart: str = "",
        git_url: str = "",
        overwrite: bool = False,
    ) -> bool:
        logger.info(f"saving model info {name}")
        res: bool = False
        model_info_: ModelInfoEntity = ModelInfoEntity()
        if overwrite:
            existed_model_: ModelInfoEntity = self.find_model(name)
            if existed_model_:
                model_info_ = existed_model_

        model_info_.name = name
        model_info_.mart = mart
        model_info_.git_url = git_url
        model_info_.saved_at = datetime.now()
        try:
            self._db_session.add(model_info_)
            self._db_session.commit()
            res = True
        except Exception as e:
            self._db_session.rollback()
            logger.exception(e)
        return res

    def update_trained_model(self, e: TrainedModelEntity) -> None:
        res: bool = False
        existing_: TrainedModelEntity = self.find_trained_model(e.file_name)

        if existing_:
            existing_.file_name = e.file_name
            existing_.train_start_dt = e.train_start_dt
            existing_.train_end_dt = e.train_end_dt
            existing_.model_data = e.model_data
            existing_.saved_at = datetime.now()

        try:
            self._db_session.add(existing_)
            self._db_session.commit()
            res = True
        except Exception as e:
            self._db_session.rollback()
            logger.exception(e)

    def _save_trained_models(
        self,
        model_name: str,
        trained_models: List[TrainedModelEntity],
        create_model_info: bool,
    ) -> bool:
        res: bool = False
        if create_model_info:
            self.save_model_info(model_name)
        model_info_: ModelInfoEntity = self.find_model(model_name)
        if model_info_:
            try:
                model_info_.trained_models.extend(trained_models)
                self._db_session.commit()
                res = True
            except Exception as e:
                self._db_session.rollback()
                logger.error(e)

        return res

    def save_trained_model(
        self, trained_model_file: str, create_model_info: bool = True
    ) -> bool:
        logger.info(f"saving trained model {trained_model_file}")
        res: bool = False
        base_name: str = basename(trained_model_file)
        fname_: str = splitext(base_name)[0]
        info_: ModelInfo = ModelInfo.from_str(fname_)
        trained_model_: TrainedModelEntity = TrainedModelEntity()
        trained_model_.file_name = fname_
        trained_model_.train_start_dt = info_.training_period.start_dt
        trained_model_.train_end_dt = info_.training_period.end_dt
        trained_model_.saved_at = datetime.now()
        with open(trained_model_file, "rb") as model_file:
            trained_model_.model_data = model_file.read()
        return self._save_trained_models(
            info_.prefix, [trained_model_], create_model_info
        )

    def save_trained_models(
        self, trained_models: List[str], create_model_info: bool = True
    ) -> bool:
        models: Dict[str, List[TrainedModelEntity]] = defaultdict(list)
        for f_ in trained_models:
            fname_: str = splitext(f_)[0]
            info_: ModelInfo = ModelInfo.from_str(fname_)
            trained_model_: TrainedModelEntity = TrainedModelEntity()
            trained_model_.file_name = fname_
            trained_model_.train_start_dt = info_.training_period.start_dt
            trained_model_.train_end_dt = info_.training_period.end_dt
            trained_model_.saved_at = datetime.now()
            with open(f_, "rb") as model_file:
                trained_model_.model_data = model_file.read()
            models[info_.prefix].append(trained_model_)

        return {
            p: self._save_trained_models(p, models[p], create_model_info)
            for p in models
        }

    def find_prediction_data(self, name: str, from_dt: datetime, to: datetime):
        pass

    def find_portfolio(self, name: str, portfolio_dt: datetime):
        pass

    def find_ground_truth(self, name: str, from_dt: datetime, to: datetime):
        pass

    def save_backtest(self, backtest_info: BackTestInfoEntity) -> bool:
        logger.info(f"saving backtest info")
        res: bool = False
        backtest_info.saved_at = datetime.now()
        try:
            self._db_session.add(backtest_info)
            self._db_session.commit()
            res = True
        except Exception as e:
            self._db_session.rollback()
            logger.exception(e)
        return res

    def find_backtest(self, tag: str = "") -> List[BackTestInfoEntity]:
        rs = self._db_session.query(BackTestInfoEntity)
        if tag:
            rs = rs.filter(BackTestInfoEntity.tag.like(f"%{tag}%")).all()
        return [c for c in rs]

    def save_portfolio(
        name: str, portfolio_dt: datetime, portfolio_data: Dict[str, pd.DataFrame]
    ):
        folio_entity = None
        rs = (
            self._db_session.query(PortfolioEntity)
            .filter(PortfolioEntity.model_info.has(ModelInfoEntity.name == name))
            .filter(PortfolioEntity.portfolio_dt == portfolio_dt)
        )

        rs_list = [c for c in rs]
        if len(rs_list) > 0:
            folio_entity = rs_list[0]

        if not folio_entity:
            folio_entity = PortfolioEntity()
            folio_entity.portfolio_dt = portfolio_dt
            model_info = self.find_model(name)
            if not model_info:
                self.save_model_info(name)

        model_info_: ModelInfoEntity = self.find_model(model_name)
        if model_info_:
            try:
                model_info_.trained_models.extend(trained_models)
                self._db_session.commit()
                res = True
            except Exception as e:
                self._db_session.rollback()
                logger.error(e)


def init_engine(db_url):
    db_engine = create_engine(db_url)
    session_maker = sessionmaker()
    session_maker.configure(bind=db_engine)
    db_session = session_maker()
    return db_engine, db_session
