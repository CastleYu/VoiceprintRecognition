from typing import TypeVar, Type, Generic, Optional, List

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import scoped_session, sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool

Base = declarative_base()
T = TypeVar('T', bound=Base)


class DAO:
    def connect(self, *args, **kwargs):
        raise NotImplementedError

    def disconnect(self, *args, **kwargs):
        raise NotImplementedError

    def commit(self):
        raise NotImplementedError

    # Create
    def add(self, *args, **kwargs):
        raise NotImplementedError

    def create(self, *args, **kwargs):
        return self.add(*args, **kwargs)

    def insert(self, *args, **kwargs):
        return self.add(*args, **kwargs)

    # Read
    def get(self, *args, **kwargs):
        raise NotImplementedError

    def search(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    def query(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    def select(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    def read(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    # Update
    def update(self, *args, **kwargs):
        raise NotImplementedError

    def alter(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def edit(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def modify(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    # Delete
    def delete(self, *args, **kwargs):
        raise NotImplementedError

    def remove(self, *args, **kwargs):
        return self.delete(*args, **kwargs)

    # Other operations
    def get_all(self):
        raise NotImplementedError

    def get_one(self, *args, **kwargs):
        raise NotImplementedError

    def delete_all(self):
        raise NotImplementedError

    def execute(self, sql: str):
        raise NotImplementedError


def sql_suppress(func):
    """装饰器：统一捕获SQLAlchemyError，回滚并记录异常信息"""

    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except SQLAlchemyError as e:
            self.session.rollback()
            print(f"[SQLAlchemyError] {func.__name__} 错误: {e}")
            return None

    return wrapper


class MySQLDAO(DAO, Generic[T]):
    def __init__(self, model: Type[T]):
        self.model: Type[T] = model

    def connect(self, host: str, port: int, user: str, password: str, database: str,
                poolclass=QueuePool,
                pool_size: int = 5,
                max_overflow: int = 10,
                pool_timeout: int = 30,
                pool_recycle: int = 600,
                echo: bool = False) -> "MySQLDAO[T]":
        DB_URL = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        self.engine = create_engine(DB_URL,
                                    poolclass=poolclass,
                                    pool_size=pool_size,
                                    max_overflow=max_overflow,
                                    pool_timeout=pool_timeout,
                                    pool_recycle=pool_recycle,
                                    echo=echo)
        Base.metadata.create_all(bind=self.engine)
        session_factory = sessionmaker(bind=self.engine)
        self._get_session = scoped_session(session_factory)
        self.session = self._get_session()
        return self

    def renew_session(self) -> "MySQLDAO[T]":
        self.session.close()
        self.session = self._get_session()
        return self

    @property
    def is_connected(self) -> bool:
        return (hasattr(self, "engine") and self.engine is not None and
                hasattr(self, "get_session") and self._get_session is not None)

    @sql_suppress
    def get(self, id_: Optional[int] = None, **kwargs) -> List[T]:
        if id_ is not None:
            entries = self.session.query(self.model).filter_by(id=id_, **kwargs).all()
        else:
            entries = self.session.query(self.model).filter_by(**kwargs).all()
        return entries

    @sql_suppress
    def get_one(self, id_: Optional[int] = None, **kwargs) -> Optional[T]:
        if id_ is not None:
            entry = self.session.query(self.model).filter_by(id=id_, **kwargs).first()
        else:
            entry = self.session.query(self.model).filter_by(**kwargs).first()
        return entry

    @sql_suppress
    def get_all(self) -> List[T]:
        return self.session.query(self.model).all()

    @sql_suppress
    def delete(self, entry_obj: T, ) -> bool:
        if not entry_obj:
            print("[SQLAlchemyError] delete 错误：目标不存在")
            return False
        self.session.delete(entry_obj)
        self.session.commit()
        return True

    @sql_suppress
    def update(self, entry_obj: T, **kwargs) -> bool:
        set_flag = True
        for key, value in kwargs.items():
            if hasattr(entry_obj, key):
                setattr(entry_obj, key, value)
            else:
                print(f"属性 {key} 不存在于对象 {entry_obj}")
                set_flag = False
        self.session.commit()
        return set_flag

    @sql_suppress
    def add(self, entry_obj: Optional[T] = None, **kwargs) -> bool:
        if entry_obj is None:
            entry_obj = self.model(**kwargs)
        self.session.add(entry_obj)
        self.session.commit()
        return True

    @sql_suppress
    def commit(self):
        self.session.commit()
        return self
