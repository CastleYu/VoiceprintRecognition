import csv
import time
import numpy as np
import pymysql
from dao.milvus_dao import MilvusClient
from utils.audioU import pre_process
from audio.vector import PaddleSpeakerVerification, SpeakerVerificationAdapter
import config
from config import AUDIO_TABLE

# 初始化Milvus客户端
milvus_client = MilvusClient(config.Milvus.host, config.Milvus.port)
milvus_client.create_collection(AUDIO_TABLE)

# 用户数据全量缓存
class UserCache:
    def __init__(self):
        self.cache = self._load_users()

    def _load_users(self):
        """一次性加载所有用户数据"""
        conn = pymysql.connect(
            host=config.MySQL.host,
            port=config.MySQL.port,
            user=config.MySQL.user,
            password=config.MySQL.password,
            database=config.MySQL.database,
            cursorclass=pymysql.cursors.DictCursor
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT voiceprint, username FROM user")
                return {str(row['voiceprint']): row['username'] for row in cursor.fetchall()}
        finally:
            conn.close()

    def get_username(self, user_id):
        return self.cache.get(user_id, 'Unknown')

# 初始化全局缓存
user_cache = UserCache()

# 初始化语音特征提取器
paddleVector = SpeakerVerificationAdapter(PaddleSpeakerVerification())

def batch_voice_recognition(test_csv_path, result_csv_path, top_k=1):
    """优化版批量语音识别函数"""
    start_time = time.time()

    with open(test_csv_path, 'r', encoding='utf-8') as f:
        test_files = [row[0] for row in csv.reader(f)][1:]

    with open(result_csv_path, 'w', newline='', encoding='utf-8') as result_file:
        writer = csv.writer(result_file)
        writer.writerow(['audio_path', 'username', 'similarity_score', 'processing_time'])

        for file_idx, file_path in enumerate(test_files, 1):
            file_start = time.time()
            try:
                # 特征提取
                processed_path = pre_process(file_path)
                audio_embedding = paddleVector.get_embedding_from_file(processed_path)

                # 向量搜索
                search_results = milvus_client.search(audio_embedding, AUDIO_TABLE, top_k)
                processing_time = time.time() - file_start

                # 直接使用缓存获取用户名
                user_ids = [str(result.id) for result in search_results[0]]
                username_map = {uid: user_cache.get_username(uid) for uid in user_ids}

                # 写入结果
                for result in search_results[0]:
                    user_id = str(result.id)
                    similarity = paddleVector.get_embeddings_score(
                        np.array(result.entity.vec, dtype=np.float32),
                        audio_embedding
                    )

                    writer.writerow([
                        file_path,
                        username_map.get(user_id, 'Unknown'),
                        f"{similarity:.4f}",
                        f"{processing_time:.4f}"
                    ])

                print(f"处理进度: {file_idx}/{len(test_files)} | 文件: {file_path}")

            except Exception as e:
                print(f"处理失败: {file_path} | 错误: {str(e)}")
                writer.writerow([file_path, 'ERROR', 'ERROR', '0.0000'])

    total_time = time.time() - start_time
    print(f"\n处理完成 | 总文件: {len(test_files)} | 耗时: {total_time:.2f}秒")
    return total_time

# 使用示例
if __name__ == "__main__":
    batch_voice_recognition(
        test_csv_path=r'P:\xiangmu\python\Voice\test_new_001.csv',
        result_csv_path=r'P:\xiangmu\python\Voice\optimized_result.csv',
        top_k=60
    )
