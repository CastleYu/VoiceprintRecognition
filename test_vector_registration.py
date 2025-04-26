import os
import json
import traceback
import numpy as np
import config
from dao import *
from config import AUDIO_TABLE, UPLOAD_FOLDER
from dao import MySQLClient, MilvusClient
from audio.vector import PaddleSpeakerVerification, SpeakerVerificationAdapter, DeepSpeakerVerification
from utils.audioU import pre_process
from utils.fileU import save_file
from utils.responseU import QuickResponse as qr

# 初始化配置
ACCURACY_THRESHOLD = 0.7  # 测试用阈值
MODELS_DIR = "./models"   # 测试用模型目录

# 初始化客户端
milvus_client = MilvusClient(config.Milvus.host, config.Milvus.port)
mysql_client = MySQLClient(config.MySQL.host, config.MySQL.port, config.MySQL.user,
                           config.MySQL.password, config.MySQL.database)

# 初始化向量模型
paddleVector = SpeakerVerificationAdapter(PaddleSpeakerVerification())
deep_speakerVector = SpeakerVerificationAdapter(DeepSpeakerVerification())

def test_vector_registration(audio_file_path, username="test_user", permission_level=1):
    """
    测试双模型向量注册功能
    :param audio_file_path: 音频文件路径
    :param username: 测试用户名
    :param permission_level: 权限级别
    :return: 注册结果
    """
    try:
        file_path = audio_file_path

        # 预处理音频文件
        pro_path = pre_process(file_path)

        print(f"Processing audio file: {file_path}")

        # 生成192维向量
        emb_192 = paddleVector.get_embedding_from_file(pro_path)
        print(f"Generated 192-dim vector: shape {emb_192.shape}")

        # 生成512维向量
        emb_512 = deep_speakerVector.get_embedding_from_file(pro_path)
        emb_512 = emb_512.squeeze()
        print(f"Generated 512-dim vector: shape {emb_512.shape}")

        # 计算平均向量 (这里保持单文件测试，实际可以扩展为多文件)
        average_192 = emb_192
        average_512 = emb_512

        # 将192维特征向量插入Milvus
        milvus_ids = milvus_client.insert(AUDIO_TABLE, [average_192.tolist()])
        print(f"Inserted 192-dim vector to Milvus with ID: {milvus_ids[0]}")

        # 将512维特征向量插入到deepSpeaker_vp512集合
        milvus_client.insert_with_ids("deepSpeaker_vp512",  [milvus_ids[0]], [average_512.tolist()])
        print(f"Inserted 512-dim vector to deepSpeaker_vp512 with same ID")
        new_user = User(username=username, voiceprint=milvus_ids[0], permission_level=permission_level)
        mysql_client.user.add(new_user)


        # 构建响应
        response = qr.data(
            voiceprint=milvus_ids[0],
            permission_level=permission_level
        )

        return response

    except Exception as e:
        traceback.print_exc()
        return qr.error(e)

    finally:
        # 清理临时文件
        pass

if __name__ == '__main__':
    # 测试用例
    test_audio_file = r"P:\xiangmu\python\Voice\VoiceprintRecognition\注册.wav"  # 替换为实际测试音频文件路径
    result = test_vector_registration(test_audio_file)
    print("\nTest Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))