import numpy as np
import config
from dao import MilvusClient

# 初始化 Milvus 客户端
mc = MilvusClient(config.Milvus.host, config.Milvus.port)

# 准备集合：192维用于自动分配 ID，512维显式插入
mc.get_dao("audio", auto_id=True, dim=192)
mc.get_dao("deepSpeaker_vp512", auto_id=False, dim=512)

# 生成随机向量
vec192 = np.random.rand(192).astype(np.float32).tolist()
vec512 = np.random.rand(512).astype(np.float32).tolist()

# 插入 192维向量，返回自动生成的 ID 列表
ids = mc.insert("audio", [vec192])

# 插入 512维向量，使用相同 ID
mc.insert_with_ids("deepSpeaker_vp512", ids, [vec512])
