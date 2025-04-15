import os
import csv
import time
from pydub import AudioSegment
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import config
from action.action_matcher import *
from audio.asr import PaddleSpeechRecognition, SpeechRecognitionAdapter
from audio.vector import PaddleSpeakerVerification, SpeakerVerificationAdapter
from config import AUDIO_TABLE
from dao.milvus_dao import MilvusClient
from dao.mysql_dao import MySQLClient
from utils.audioU import pre_process

# ACCURACY_THRESHOLD = config.Algorithm.threshold#0.8
ACCURACY_THRESHOLD = 0.2
MODELS_DIR = config.Update.ModelDir

milvus_client = MilvusClient(config.Milvus.host, config.Milvus.port)
milvus_client.create_collection(AUDIO_TABLE)
mysql_client = MySQLClient(config.MySQL.host, config.MySQL.port, config.MySQL.user,
                           config.MySQL.password, config.MySQL.database)

paddleASR = SpeechRecognitionAdapter(PaddleSpeechRecognition())
paddleVector = SpeakerVerificationAdapter(PaddleSpeakerVerification())

action_matcher = InstructionMatcher(MODELS_DIR).load(MicomlMatcher('paraphrase-multilingual-MiniLM-L12-v2'))


def do_search_action(action):
    action_set = mysql_client.get_all_actions()
    best_match, similarity_score = action_matcher.match(action, action_set)
    action_id = mysql_client.get_action_id(best_match)
    return action_id, best_match, similarity_score

class BatchProcessor:
    def __init__(self):
        self.milvus_client = MilvusClient(config.Milvus.host, config.Milvus.port)
        self.mysql_client = MySQLClient(
            config.MySQL.host, config.MySQL.port,
            config.MySQL.user, config.MySQL.password,
            config.MySQL.database
        )
        self.vector_engine = SpeakerVerificationAdapter(PaddleSpeakerVerification())
        self.asr_engine = paddleASR  # 假设已初始化

        # 初始化输出目录
        self.output_dir = Path("batch_results")
        self.output_dir.mkdir(exist_ok=True)

    def process_batch(self, csv_path: str):
        """批量处理入口方法"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 初始化结果文件
        success_file = self.output_dir / f"success_{timestamp}.csv"
        fail_file = self.output_dir / f"fail_{timestamp}.csv"
        time_log_file = self.output_dir / f"timing_{timestamp}.csv"

        with open(csv_path, 'r') as input_f, \
             open(success_file, 'w', newline='') as success_f, \
             open(fail_file, 'w', newline='') as fail_f, \
             open(time_log_file, 'w', newline='') as time_f:

            # 初始化写入器
            success_writer = csv.DictWriter(success_f, fieldnames=self._get_success_header())
            fail_writer = csv.DictWriter(fail_f, fieldnames=self._get_fail_header())
            time_writer = csv.DictWriter(time_f, fieldnames=self._get_timing_header())

            success_writer.writeheader()
            fail_writer.writeheader()
            time_writer.writeheader()

            reader = csv.DictReader(input_f)
            for row_idx, row in enumerate(reader, 1):
                file_path = row['filepath']
                print(f"\nProcessing {row_idx}: {file_path}")

                # 初始化计时数据
                timing = {'filepath': file_path}
                start_total = time.perf_counter()

                try:
                    result = self._process_single_file(file_path, timing)
                except Exception as e:
                    result = self._handle_error(e, file_path)

                # 记录总耗时
                timing['total'] = time.perf_counter() - start_total

                # 写入结果
                self._write_results(
                    success_writer,
                    fail_writer,
                    time_writer,
                    file_path,
                    result,
                    timing
                )

                # 实时刷新写入
                for f in [success_f, fail_f, time_f]:
                    f.flush()

        print(f"\nBatch processing completed. Results saved to: {self.output_dir}")



    def _process_single_file(self, file_path: str, timing: dict) -> dict:
        """处理单个文件的核心逻辑"""
        result = {'status': 'success'}
        try:
            # 预处理阶段
            start = time.perf_counter()
            processed_path = pre_process(file_path)

            timing['preprocess'] = time.perf_counter() - start

            # 特征提取
            start = time.perf_counter()
            embedding = self.vector_engine.get_embedding(processed_path)
            timing['feature_extract'] = time.perf_counter() - start

            # 向量搜索
            start = time.perf_counter()
            search_results = self.milvus_client.search(embedding, AUDIO_TABLE, 1)
            timing['milvus_search'] = time.perf_counter() - start

            if not search_results:
                raise ValueError("No matching voiceprint found")

            # 解析搜索结果
            top_match = search_results[0][0]
            user_id = str(top_match.id)
            similarity = top_match.distance

            # 相似度计算
            start = time.perf_counter()
            similarity_score = self.vector_engine.get_embeddings_score(
                np.array(top_match.entity.vec, dtype=np.float32),
                embedding
            )
            timing['similarity_calc'] = time.perf_counter() - start

            if similarity_score < ACCURACY_THRESHOLD:
                raise PermissionError(f"Similarity score {similarity_score:.2f} below threshold")

            # 数据库查询
            start = time.perf_counter()
            username = self.mysql_client.find_user_name_by_id(user_id)
            permission = self.mysql_client.find_permission_level_by_id(user_id)
            timing['mysql_query'] = time.perf_counter() - start

            # # 语音识别
            # start = time.perf_counter()
            # # asr_text = self.asr_engine.recognize(file_path)
            # # action = do_search_action(asr_text)[1]
            # timing['asr_process'] = time.perf_counter() - start

            # 组装结果
            result.update({
                'user_id': user_id,
                'username': username,
                'permission': permission,
                'similarity': f"{similarity_score:.4f}",
                # 'asr_text': asr_text,
                # 'action': action
            })

        except Exception as e:
            result = self._handle_error(e, file_path)
        finally:
            pass

        return result



    def _handle_error(self, error: Exception, filepath: str) -> dict:
        """错误处理"""
        error_type = type(error).__name__
        return {
            'status': 'fail',
            'error_type': error_type,
            'error_msg': str(error),
            'stack_trace': traceback.format_exc(limit=3)
        }

    def _write_results(self, success_writer, fail_writer, time_writer, filepath, result, timing):
        """结果写入方法"""
        # 写入时间日志
        time_writer.writerow(timing)

        # 写入业务结果
        if result['status'] == 'success':
            success_writer.writerow({
                'filepath': filepath,
                'user_id': result.get('user_id', ''),
                'username': result.get('username', ''),
                'permission': result.get('permission', ''),
                'similarity': result.get('similarity', ''),
                # 'asr_text': result.get('asr_text', ''),
                # 'action': result.get('action', '')
            })
        else:
            fail_writer.writerow({
                'filepath': filepath,
                'error_type': result.get('error_type', ''),
                'error_msg': result.get('error_msg', ''),
                'similarity': result.get('similarity', ''),
                'stack_trace': result.get('stack_trace', '')
            })

    def _get_success_header(self):
        # return ['filepath', 'user_id', 'username', 'permission', 'similarity', 'asr_text', 'action']
        return ['filepath', 'user_id', 'username', 'permission', 'similarity']

    def _get_fail_header(self):
        return ['filepath', 'error_type', 'error_msg', 'similarity', 'stack_trace']

    def _get_timing_header(self):
        # return [
        #     'filepath', 'preprocess', 'feature_extract', 'milvus_search',
        #     'similarity_calc', 'mysql_query', 'asr_process', 'total'
        # ]
        return [
            'filepath', 'preprocess', 'feature_extract', 'milvus_search',
            'similarity_calc', 'mysql_query',  'total'
        ]

    def _cleanup_temp_files(self, files):
        """清理临时文件"""
        for f in files:
            try:
                if Path(f).exists():
                    os.remove(f)
            except Exception as e:
                print(f"Error deleting temp file {f}: {str(e)}")


if __name__ == "__main__":
    processor = BatchProcessor()
    processor.process_batch(r"P:\xiangmu\python\Voice\Test_6_7.csv")