# Milvus数据库
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility


class MilvusClient:
    def __init__(self, host, port):
        connections.connect(alias='default', host=host, port=port)

    def create_collection(self, table_name):
        # 检查集合是否已经存在
        if utility.has_collection(table_name):
            return Collection(name=table_name)

        # 定义集合的 schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # 主键
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=192)
        ]
        schema = CollectionSchema(fields, description="audio collection")

        # 创建集合
        collection = Collection(name=table_name, schema=schema)
        return collection

    def insert(self, table_name, vectors):
        # 确保集合已经创建
        collection = self.create_collection(table_name)
        # 插入数据并返回 ID
        ids = collection.insert([vectors]).primary_keys  # 获取 ID
        return ids

    def create_index(self, table_name):
        collection = Collection(name=table_name)
        index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
        collection.create_index(field_name="vec", index_params=index_params)

    def search(self, query_vectors, table_name, top_k=1):
        # 确保集合已经创建
        collection = self.create_collection(table_name)
        # 搜索
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(data=[query_vectors], anns_field="vec", param=search_params, limit=top_k,
                                    output_fields=['vec'])
        return results
