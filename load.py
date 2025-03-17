from collections import defaultdict
import csv
import time
import traceback
import numpy as np
from config import AUDIO_TABLE, USER_TABLE
from dao.milvus_dao import MilvusClient
from dao.mysql_dao import MySQLClient
from utils.audioU import pre_process
from audio.vector import PaddleSpeakerVerification, SpeakerVerificationAdapter
import config

class DataLoader:
    def __init__(self):
        self.milvus_client = MilvusClient(config.Milvus.host, config.Milvus.port)
        self.mysql_client = MySQLClient(
            config.MySQL.host, config.MySQL.port,
            config.MySQL.user, config.MySQL.password,
            config.MySQL.database
        )
        self.vector_engine = SpeakerVerificationAdapter(PaddleSpeakerVerification())

        # 初始化数据库结构
        self._initialize_databases()

    def _initialize_databases(self):
        """初始化数据库和集合"""

        # 创建MySQL表（如果不存在）
        self.mysql_client.create_mysql_table(USER_TABLE)


    def process_csv(self, csv_path):
        """处理CSV文件加载数据（添加时间统计）"""
        file_data = defaultdict(list)

        # 读取CSV文件
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) < 3:
                        continue
                    file_path, username, level = row[0], row[1], int(row[2])
                    file_data[(username, level)].append(file_path)
        except Exception as e:
            print(f"CSV文件读取失败: {str(e)}")
            return False

        total_users = len(file_data)
        for idx, ((username, level), paths) in enumerate(file_data.items(), 1):
            print(f"\n=== 处理用户 {idx}/{total_users}: {username} ===")
            total_convert = 0.0
            milvus_time = 0.0
            mysql_time = 0.0

            try:
                # 特征提取与平均
                embeddings = []
                for path in paths:
                    try:
                        # 单个文件转换计时
                        start_convert = time.perf_counter()
                        processed = pre_process(path)
                        emb = self.vector_engine.get_embedding(processed)
                        embeddings.append(emb)
                        total_convert += time.perf_counter() - start_convert
                    except Exception as e:
                        print(f"文件处理失败: {path} - {str(e)}")
                        continue

                if not embeddings:
                    print(f"用户 {username} 无有效音频")
                    continue

                # 向量平均计算
                avg_start = time.perf_counter()
                avg_embedding = np.mean(np.array(embeddings), axis=0).tolist()
                avg_time = time.perf_counter() - avg_start
                print(f"向量计算耗时: {avg_time:.3f}s")

                # 插入Milvus
                milvus_start = time.perf_counter()
                milvus_ids = self.milvus_client.insert(AUDIO_TABLE, [avg_embedding])
                milvus_time = time.perf_counter() - milvus_start
                if not milvus_ids:
                    print(f"Milvus插入失败: {username}")
                    continue

                # 插入MySQL
                mysql_start = time.perf_counter()
                user_id = self.mysql_client.load_data_to_mysql(USER_TABLE, [(username, milvus_ids[0], level)])
                mysql_time = time.perf_counter() - mysql_start

                # 输出时间统计
                print(f"[性能统计] {username}")
                print(f" 音频处理: {total_convert:.3f}s ({len(embeddings)}个文件)")
                print(f" Milvus插入: {milvus_time:.3f}s")
                print(f" MySQL插入: {mysql_time:.3f}s")
                print(f" 总耗时: {total_convert + milvus_time + mysql_time:.3f}s")

            except Exception as e:
                error_msg = f"用户 {username} 处理异常: {str(e)}"
                if 'user_id' in locals():
                    error_msg = f"用户 {user_id} 处理异常: {str(e)}"
                print(error_msg)
                traceback.print_exc()

        return True

def load():
    CSV_PATH = r'P:\xiangmu\python\Voice\Data\Train_001.csv'

    loader = DataLoader()
    if loader.process_csv(CSV_PATH):
        print("数据加载完成")
    else:
        print("数据加载失败")

if __name__ == "__main__":
    load()
