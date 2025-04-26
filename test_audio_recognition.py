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
LOW_TEXT_CHARACTER_LIMIT = 6  # 测试用低文本量阈值
HIGH_TEXT_CHARACTER_LIMIT = 12  # 测试用高文本量阈值

# 初始化客户端
milvus_client = MilvusClient(config.Milvus.host, config.Milvus.port)
mysql_client = MySQLClient(config.MySQL.host, config.MySQL.port, config.MySQL.user,
                           config.MySQL.password, config.MySQL.database)

# 初始化向量模型
paddleVector = SpeakerVerificationAdapter(PaddleSpeakerVerification())
deep_speakerVector = SpeakerVerificationAdapter(DeepSpeakerVerification())

def test_audio_recognition(audio_file_path):
    """
    测试音频识别功能
    :param audio_file_path: 音频文件路径
    :return: 识别结果
    """
    try:
        file_path = audio_file_path

        # 计算音频时长和文本量
        import librosa
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        text_estimate = duration * (110 / 60)  # 取中间值110字/min

        print(f"Processing audio file: {file_path}")
        print(f"Audio duration: {duration:.2f}s, estimated text length: {text_estimate:.1f}")

        # 预处理音频文件
        pro_path = pre_process(file_path)

        # 根据文本量决定模型组合
        use_paddle = text_estimate >= LOW_TEXT_CHARACTER_LIMIT
        use_deep = text_estimate <= HIGH_TEXT_CHARACTER_LIMIT

        print(f"Model selection - Paddle: {use_paddle}, DeepSpeaker: {use_deep}")

        # 获取音频嵌入向量
        paddle_embedding = paddleVector.get_embedding_from_file(pro_path) if use_paddle else None
        deep_embedding = deep_speakerVector.get_embedding_from_file(pro_path).squeeze() if use_deep else None

        if use_paddle:
            print(f"Generated Paddle vector: shape {paddle_embedding.shape}")
        if use_deep:
            print(f"Generated DeepSpeaker vector: shape {deep_embedding.shape}")

        # 在 Milvus 中搜索相似音频
        paddle_results = milvus_client.search(AUDIO_TABLE, paddle_embedding, top_k=1) if use_paddle else []
        deep_results = milvus_client.search("deepSpeaker_vp512", deep_embedding, top_k=1) if use_deep else []

        recognize_result = "FAILED"
        VoicePrint_id = '0'
        user_id = '0'
        similarity_score = '0'
        model_used = 'none'

        # 判断识别结果
        if use_paddle and use_deep:
            # 使用两种模型的情况
            model_used = 'both'
            paddle_match = len(paddle_results) > 0 and paddleVector.get_embeddings_score(
                np.array(paddle_results[0][0].entity.vec, dtype=np.float32), paddle_embedding) >= ACCURACY_THRESHOLD
            deep_match = len(deep_results) > 0 and deep_speakerVector.get_embeddings_score(
                np.array(deep_results[0][0].entity.vec, dtype=np.float32)[np.newaxis, :], deep_embedding[np.newaxis, :]) >= ACCURACY_THRESHOLD

            if ACCURACY_THRESHOLD < 0.8:  # 假设HIGH_PRECISION_THRESHOLD为0.8
                recognize_result = "SUCCESS" if paddle_match or deep_match else "FAILED"
            else:
                recognize_result = "SUCCESS" if paddle_match and deep_match else "FAILED"

            if recognize_result == "SUCCESS":
                VoicePrint_id = str(paddle_results[0][0].id if paddle_match else deep_results[0][0].id)
                similarity_score = max(
                    paddleVector.get_embeddings_score(
                        np.array(paddle_results[0][0].entity.vec, dtype=np.float32), paddle_embedding) if paddle_match else 0,
                    deep_speakerVector.get_embeddings_score(
                        np.array(deep_results[0][0].entity.vec, dtype=np.float32)[np.newaxis, :], deep_embedding[np.newaxis, :]) if deep_match else 0
                )
        elif use_paddle:
            # 仅使用paddle模型
            model_used = 'paddle'
            if len(paddle_results) > 0:
                similarity_score = paddleVector.get_embeddings_score(
                    np.array(paddle_results[0][0].entity.vec, dtype=np.float32), paddle_embedding)
                recognize_result = "SUCCESS" if similarity_score >= ACCURACY_THRESHOLD else "FAILED"
                VoicePrint_id = str(paddle_results[0][0].id)
        elif use_deep:
            # 仅使用deep模型
            model_used = 'deep'
            if len(deep_results) > 0:
                similarity_score = deep_speakerVector.get_embeddings_score(
                    np.array(deep_results[0][0].entity.vec, dtype=np.float32)[np.newaxis, :], deep_embedding[np.newaxis, :])
                recognize_result = "SUCCESS" if similarity_score >= ACCURACY_THRESHOLD else "FAILED"
                VoicePrint_id = str(deep_results[0][0].id)

        # 获取用户信息
        user_name = 'None'
        permission_level = 0
        if recognize_result == "SUCCESS":
            user = mysql_client.get_user_by_voiceprint('['+VoicePrint_id+']')
            if user:
                user_name = user.username
                permission_level = user.permission_level

        # 构建响应
        response = {
            "recognize_result": recognize_result,
            "username": user_name,
            "user_id": user_id,
            "permission_level": permission_level,
            "similarity_score": float(similarity_score) if similarity_score != '0' else 0,
            "model_used": model_used,
            "audio_duration": duration,
            "estimated_text_length": text_estimate
        }

        return response

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

    finally:
        # 清理临时文件
        pass

if __name__ == '__main__':
    # 测试用例
    test_audio_file = r"P:\xiangmu\python\Voice\Data\test\21_蔡培\[2025-03-03][18-41-22].wav"  # 替换为实际测试音频文件路径
    result = test_audio_recognition(test_audio_file)
    print("\nTest Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))