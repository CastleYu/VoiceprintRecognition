from typing import TypeVar, Type, Generic, Optional, List, Union

from sqlalchemy import create_engine, Pool, Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import scoped_session, sessionmaker, declarative_base, Session
from sqlalchemy.pool import QueuePool


class Schema:

    @classmethod
    def get_fields(cls):
        """
        遍历类属性，将所有 FieldSchema 实例收集到一个列表中
        """
        return [value for key, value in cls.__dict__.items() if isinstance(value, FieldSchema)]


Base = declarative_base()
T = TypeVar('T', bound=Base)


class DAOBase:
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


class SQLDAO(DAOBase, Generic[T]):
    engine: Engine
    session: Session
    _get_session: scoped_session[Session]

    def __init__(self, model: Type[T]):
        self.model: Type[T] = model

    def renew_session(self) -> "SQLDAO[T]":
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
        """
        update有两种方案，一种是直接修改ORM对象然后commit，不必走本函数；
        另一种是使用本函数，其实本身也要传入对象，要修改的字段以键值对的形式传入
        :param entry_obj:
        :param kwargs:
        :return:
        """
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
        """
        增加一个条目，可以传入一个对象，也可以传入字段键值对，（传入对象优先于键值对）
        :param entry_obj:
        :param kwargs:
        :return:
        """
        if entry_obj is None:
            entry_obj = self.model(**kwargs)
        self.session.add(entry_obj)
        self.session.commit()
        return True

    @sql_suppress
    def commit(self):
        self.session.commit()
        return self


class MySQLDAO(SQLDAO):
    """
    一个表对应一个ORM模型对应一个DAO
    """

    def connect(self, host: str, port: int, user: str, password: str, database: str,
                poolclass: Type[Pool] = QueuePool,
                pool_size: int = 5,
                max_overflow: int = 10,
                pool_timeout: int = 30,
                pool_recycle: int = 600,
                echo: bool = False) -> "MySQLDAO[T]":
        DB_URL = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        self.engine: Engine = create_engine(DB_URL,
                                            poolclass=poolclass,
                                            pool_size=pool_size,
                                            max_overflow=max_overflow,
                                            pool_timeout=pool_timeout,
                                            pool_recycle=pool_recycle,
                                            echo=echo)
        Base.metadata.create_all(bind=self.engine)
        session_factory = sessionmaker(bind=self.engine)
        self._get_session = scoped_session(session_factory)
        self.session: Session = self._get_session()
        return self


class SQLiteDAO(SQLDAO):
    def connect(self, database: str = "default.db", echo: bool = False) -> "SQLiteDAO[T]":
        DB_URL = f"sqlite:///./{database}"
        self.engine: Engine = create_engine(DB_URL, connect_args={
            "check_same_thread": False}, echo=echo)
        Base.metadata.create_all(bind=self.engine)
        session_factory = sessionmaker(bind=self.engine)
        self._get_session = scoped_session(session_factory)
        self.session: Session = self._get_session()
        return self


from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)


class MilvusDAO:
    collection: Collection

    def __init__(self, collection_name: str, dim: int = 192, index_params: dict = None):
        self.collection_name = collection_name
        self.dim = dim
        self.index_params = index_params or {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {
                "nlist": 128}
        }
        self.collection: Optional[Collection] = None

    def connect(self, host: str, port: int):
        """连接 Milvus 服务器，并获取或创建集合"""
        connections.connect(alias='default', host=host, port=port)
        self.collection = self._create_or_get_collection(self.collection_name)
        return self

    def _create_or_get_collection(self, name: str) -> Collection:
        """如果集合存在则加载，不存在则创建集合和索引"""
        if utility.has_collection(name):
            collection = Collection(name=name)
            if not collection.has_index():
                collection.create_index(field_name="vec", index_params=self.index_params)
            collection.load()
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
            ]
            schema = CollectionSchema(fields=fields, description="vector collection")
            collection = Collection(name=name, schema=schema)
            collection.create_index(field_name="vec", index_params=self.index_params)
            collection.load()
        return collection

    def add(self, vectors: list):
        """
        插入向量数据
        :param vectors: 单个向量或向量列表（示例中将数据包装在列表中）
        :return: 插入数据的主键列表
        """
        result = self.collection.insert([vectors])
        return result.primary_keys

    def search(self, query_vector: list, top_k: int = 1, nprobe: int = 10):
        """
        搜索最相近的向量
        :param query_vector: 查询向量
        :param top_k: 返回最相似的前 k 个结果
        :param nprobe: 搜索参数 nprobe（影响搜索准确性和速度）
        :return: 搜索结果列表
        """
        search_params = {
            "metric_type": "L2",
            "params": {
                "nprobe": nprobe}}
        results = self.collection.search(
            data=[query_vector],
            anns_field="vec",
            param=search_params,
            limit=top_k,
            output_fields=['vec']
        )
        return results

    def delete_by_id(self, id_to_delete: Union[int, str]) -> bool:
        """
        根据主键删除数据
        :param id_to_delete: 需要删除的 ID
        :return: 删除是否成功
        """
        self.collection.delete(f"id in [{id_to_delete}]")
        self.collection.compact()
        result = self.collection.query(f"id in [{id_to_delete}]")
        return len(result) == 0

    def delete_all(self) -> bool:
        """
        删除集合中所有数据
        :return: 如果全部删除则返回 True，否则抛出异常
        """
        self.collection.delete("id >= 0")
        self.collection.compact()
        result = self.collection.query("id >= 0")
        if len(result) != 0:
            raise RuntimeError("Milvus delete all failed")
        return True

    def get_all(self):
        """查询集合中所有数据"""
        return self.collection.query("id >= 0")
