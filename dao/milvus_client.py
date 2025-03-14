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

    def search(self, collection_name: str, query_vectors: list, top_k: int = 1, nprobe: int = 10):
        dao = self.get_dao(collection_name)
        return dao.search(query_vectors, top_k=top_k, nprobe=nprobe)

    def delete_by_id(self, collection_name: str, id_to_delete: int) -> bool:
        dao = self.get_dao(collection_name)
        return dao.delete_by_id(id_to_delete)

    def delete_all(self, collection_name: str) -> bool:
        dao = self.get_dao(collection_name)
        return dao.delete_all()

    def query_all(self, collection_name: str):
        dao = self.get_dao(collection_name)
        return dao.get_all()


# ===== 示例 =====
if __name__ == "__main__":
    # 初始化 Milvus 客户端（请替换为实际的主机和端口）
    client = MilvusClient(host="127.0.0.1", port=19530)
    collection = "audio_collection"

    # 示例：插入数据（向量数据需要根据实际维度提供）
    vectors = [0.1] * 192  # 示例向量，192 维
    ids = client.insert(collection, vectors)
    print("插入数据的主键:", ids)

    # 示例：搜索
    results = client.search(collection, vectors, top_k=3)
    print("搜索结果:", results)

    # 示例：删除指定数据
    if ids:
        success = client.delete_by_id(collection, ids[0])
        print("删除操作成功:", success)

    # 示例：查询所有数据
    all_data = client.query_all(collection)
    print("所有数据:", all_data)
