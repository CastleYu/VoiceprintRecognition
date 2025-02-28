from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility


class MilvusClient:
    def __init__(self, host, port):
        connections.connect(alias='default', host=host, port=port)
        self.fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=192)
        ]
        self.index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {
                "nlist": 128}
        }

    def create_collection(self, table_name):
        if utility.has_collection(table_name):
            collection = Collection(name=table_name)
            # 确保已有集合存在索引，若不存在则创建
            if not collection.has_index():
                collection.create_index(field_name="vec", index_params=self.index_params)
            collection.load()
            return collection

        # 创建新集合并设置Schema
        schema = CollectionSchema(self.fields, description="audio collection")
        collection = Collection(name=table_name, schema=schema)
        # 创建索引
        collection.create_index(field_name="vec", index_params=self.index_params)
        # 加载集合到内存
        collection.load()
        return collection

    def insert(self, table_name, vectors):
        collection = self.create_collection(table_name)
        # 插入数据并返回ID
        ids = collection.insert([vectors]).primary_keys
        return ids

    def create_index(self, table_name):
        collection = Collection(name=table_name)
        collection.create_index(field_name="vec", index_params=self.index_params)

    def search(self, query_vectors, table_name, top_k=1):
        # 确保集合已经创建
        collection = self.create_collection(table_name)
        # 搜索
        search_params = {
            "metric_type": "L2",
            "params": {
                "nprobe": 10}}
        results = collection.search(data=[query_vectors], anns_field="vec", param=search_params, limit=top_k,
                                    output_fields=['vec'])
        return results

    def delete_by_id(self, table_name, id_to_delete):
        # 确保集合已经创建
        collection = self.create_collection(table_name)
        # 删除指定 ID 的数据
        collection.delete(f"id in [{id_to_delete}]")
        collection.compact()
        # 检查删除是否成功
        result = collection.query(f"id in [{id_to_delete}]")
        return len(result) == 0  # 如果返回空列表，表示删除成功

    def delete_all(self, table_name):
        # 确保集合已经创建
        collection = self.create_collection(table_name)
        # 删除集合中的所有数据
        collection.delete("id >= 0")
        collection.compact()
        # 检查是否已删除所有数据
        result = collection.query("id >= 0")
        if len(result) == 0:  # 如果返回空列表，表示所有数据已删除
            return
        else:
            print(result)
            raise RuntimeError("milvus delete all failed")

    def query_all(self, table_name):
        collection = self.create_collection(table_name)
        result = collection.query("id >= 0")
        print(result)
