from typing import Union

from ._dao import MilvusDAO


class MilvusClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.daos = {}  # 保存各集合对应的 DAO 对象

    def get_dao(self, collection_name: str) -> MilvusDAO:
        """
        根据集合名称获取对应的 DAO 实例；若不存在则创建并连接
        """
        if collection_name not in self.daos:
            dao = MilvusDAO(collection_name=collection_name)
            dao.connect(self.host, self.port)
            self.daos[collection_name] = dao
        return self.daos[collection_name]

    def insert(self, collection_name: str, vectors: list):
        dao = self.get_dao(collection_name)
        return dao.add(vectors)

    def search(self, collection_name: str, query_vector: list, top_k: int = 1, nprobe: int = 10):
        dao = self.get_dao(collection_name)
        return dao.search(query_vector, top_k=top_k, nprobe=nprobe)

    def delete_by_id(self, collection_name: str, id_to_delete: Union[int, str]) -> bool:
        dao = self.get_dao(collection_name)
        return dao.delete_by_id(id_to_delete)

    def delete_all(self, collection_name: str) -> bool:
        dao = self.get_dao(collection_name)
        return dao.delete_all()

    def query_all(self, collection_name: str):
        dao = self.get_dao(collection_name)
        return dao.get_all()
