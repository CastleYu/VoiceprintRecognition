import os
import csv
import time
from pydub import AudioSegment
from collections import defaultdict
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
import config
from action.action_matcher import *
from audio.asr import PaddleSpeechRecognition, SpeechRecognitionAdapter
from audio.vector import PaddleSpeakerVerification, SpeakerVerificationAdapter
from config import AUDIO_TABLE, USER_TABLE
from dao.milvus_dao import MilvusClient
from dao.mysql_dao import MySQLClient
from utils.audioU import pre_process

ACCURACY_THRESHOLD = config.Algorithm.threshold
MODELS_DIR = config.Update.ModelDir

milvus_client = MilvusClient(config.Milvus.host, config.Milvus.port)
milvus_client.create_collection(AUDIO_TABLE)
mysql_client = MySQLClient(config.MySQL.host, config.MySQL.port, config.MySQL.user,
                           config.MySQL.password, config.MySQL.database)

paddleASR = SpeechRecognitionAdapter(PaddleSpeechRecognition())
paddleVector = SpeakerVerificationAdapter(PaddleSpeakerVerification())

action_matcher = InstructionMatcher(MODELS_DIR).load(MicomlMatcher('paraphrase-multilingual-MiniLM-L12-v2'))

def concatenate_audio_files(file_list):
    """
    将音频文件列表按顺序拼接成新文件，自动生成输出路径
    参数：
        file_list: 要拼接的音频文件路径列表
    返回：
        成功时返回新文件路径，失败时返回None
    """
    # 有效性检查
    if not file_list:
        print("🚫 文件列表为空")
        return None

    # 创建空音频容器
    combined = AudioSegment.empty()
    valid_count = 0

    try:
        # 遍历处理每个文件
        for idx, file_path in enumerate(file_list):
            try:
                # 加载音频文件
                audio = AudioSegment.from_file(file_path)
                combined += audio
                valid_count += 1

                # 记录第一个有效文件的目录信息
                if valid_count == 1:
                    first_file_dir = os.path.dirname(file_path)
                    first_file_name = os.path.basename(file_path)
            except Exception as e:
                print(f"⚠️ 跳过无效文件 [{idx+1}/{len(file_list)}]: {file_path}")
                print(f"错误详情: {str(e)}")

        # 有效性检查
        if valid_count == 0:
            print("🚫 没有有效音频文件可拼接")
            return None

        # 生成智能文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"combined_{timestamp}"

        # 保持与第一个文件相同的格式
        ext = os.path.splitext(first_file_name)[-1].lower()
        output_file = os.path.join(first_file_dir, f"{base_name}{ext}")

        # 防重复处理
        counter = 1
        while os.path.exists(output_file):
            output_file = os.path.join(first_file_dir, f"{base_name}_{counter}{ext}")
            counter += 1

        # 导出音频文件
        combined.export(output_file, format=ext[1:])  # 去除扩展名前的点

        # 打印处理报告
        print(f"✅ 成功拼接 {valid_count}/{len(file_list)} 个文件")
        print(f"📁 输出文件: {output_file}")
        print(f"⏱️ 总时长: {len(combined)/1000:.2f}秒")

        return output_file

    except Exception as e:
        print(f"❌ 严重错误: {str(e)}")
        return None


def batch_load(csv_path: str):
    try:
        start_warmup = time.perf_counter()  # 新增开始计时
        mysql_client.get_all_users()  # 原有预热操作
    except Exception as e:
        warmup_duration = time.perf_counter() - start_warmup  # 计算耗时
        print(f"数据库预热失败（不影响后续操作）[耗时 {warmup_duration:.2f}s]: {str(e)}")  # 新增耗时显示
        traceback.print_exc()
    else:
        warmup_duration = time.perf_counter() - start_warmup  # 成功耗时计算
        print(f"数据库预热成功 [耗时 {warmup_duration:.2f}s]")  # 新增成功耗时
    """批量处理CSV文件中的声纹注册请求"""
     # 创建结果目录
    output_dir = Path("batch_registration_results")
    output_dir.mkdir(exist_ok=True)

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 初始化日志文件
    success_file = output_dir / f"success_{timestamp}.csv"
    fail_file = output_dir / f"fail_{timestamp}.csv"
    time_log_file = output_dir / f"timing_{timestamp}.csv"

    with open(csv_path, 'r') as input_f, \
         open(success_file, 'w', newline='') as success_f, \
         open(fail_file, 'w', newline='') as fail_f, \
         open(time_log_file, 'w', newline='') as time_f:

        # 初始化增强版写入器
        success_writer = csv.DictWriter(success_f, fieldnames=[
            'directory', 'file_count', 'username', 'permission_level',
            'user_id', 'voiceprint_id', 'processed_files'
        ])

        fail_writer = csv.DictWriter(fail_f, fieldnames=[
            'directory', 'file_count', 'username', 'error_type',
            'error_msg', 'failed_files'
        ])

        time_writer = csv.DictWriter(time_f, fieldnames=[
            'directory', 'preprocess', 'feature_extract', 'vector_avg',
            'milvus_insert', 'mysql_insert', 'total'
        ])

        # 写入表头
        success_writer.writeheader()
        fail_writer.writeheader()
        time_writer.writeheader()

        # 构建目录分组
        reader = csv.DictReader(input_f)
        dir_groups = defaultdict(list)
        for row in reader:
            dir_path = Path(row['filename']).parent
            dir_groups[dir_path].append(row)

        # 处理每个目录组
        for dir_idx, (dir_path, rows) in enumerate(dir_groups.items(), 1):
            print(f"\n📂 正在处理目录 {dir_idx}/{len(dir_groups)}: {dir_path}")

            # 准备基础数据
            file_list = [row['filename'] for row in rows]
            username = rows[0]['username']
            permission_level = int(rows[0]['permission_level'])
            file_count = len(rows)

            # 初始化计时
            timing = {'directory': str(dir_path)}
            start_total = time.perf_counter()

            try:
                # 验证数据一致性
                if any(row['username'] != username for row in rows):
                    raise ValueError("目录内存在不一致的用户名")
                if any(int(row['permission_level']) != permission_level for row in rows):
                    raise ValueError("目录内存在不一致的权限等级")

                # 执行批量注册
                result, timing = process_registration(
                    file_list=file_list,
                    username=username,
                    permission_level=permission_level,
                    timing=timing
                )

                # 写入成功记录
                success_writer.writerow({
                    'directory': str(dir_path),
                    'file_count': file_count,
                    'username': username,
                    'permission_level': permission_level,
                    'user_id': result['user_id'],
                    'voiceprint_id': result['voiceprint_id'],
                    'processed_files': ';'.join(file_list)
                })

            except Exception as e:
                # 写入失败记录
                fail_writer.writerow({
                    'directory': str(dir_path),
                    'file_count': file_count,
                    'username': username,
                    'error_type': type(e).__name__,
                    'error_msg': str(e),
                    'failed_files': ';'.join(file_list)
                })
                traceback.print_exc()
            finally:
                # 记录时间日志
                timing['total'] = time.perf_counter() - start_total
                time_writer.writerow(timing)

                # 实时刷新写入
                success_f.flush()
                fail_f.flush()
                time_f.flush()

            time.sleep(1)  # 每个文件处理完暂停

    print(f"\n✅ 批量处理完成！结果保存在: {output_dir}")

def process_registration(file_list , username: str, permission_level: int, timing: dict) -> tuple:
    """处理单个注册请求"""
    result = {}
    temp_files = []

    try:
        # === 文件预处理 ===
        start = time.perf_counter()
        processed_path = concatenate_audio_files(file_list)
        processed_path = pre_process(processed_path)
        temp_files.append(processed_path)
        timing['preprocess'] = time.perf_counter() - start

        # === 特征提取 ===
        start = time.perf_counter()
        audio_emb = paddleVector.get_embedding(processed_path)
        timing['feature_extract'] = time.perf_counter() - start

        # === 多文件向量平均 ===
        start = time.perf_counter()
        avg_embedding = np.mean([audio_emb], axis=0)
        timing['vector_avg'] = time.perf_counter() - start

        # === Milvus 操作 ===
        start = time.perf_counter()
        milvus_ids = milvus_client.insert(AUDIO_TABLE, [avg_embedding.tolist()])
        milvus_client.create_index(AUDIO_TABLE)
        timing['milvus_insert'] = time.perf_counter() - start

        # === MySQL 操作 ===
        start = time.perf_counter()
        user_id = mysql_client.load_data_to_mysql(
            USER_TABLE,
            [(username, milvus_ids[0], permission_level)]
        )
        timing['mysql_insert'] = time.perf_counter() - start

        # 组装结果
        result = {
            'user_id': user_id,
            'voiceprint_id': milvus_ids[0]
        }

    except Exception as e:
        raise RegistrationError(f"处理失败: {str(e)}") from e

    finally:
        # 清理临时文件
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"清理文件失败: {str(e)}")

    return result, timing

class RegistrationError(Exception):
    """自定义注册异常"""
    pass

if __name__ == "__main__":
    batch_load(r"P:\xiangmu\python\Voice\Data\Train_003.csv")
