import os
import json
import traceback
import numpy as np
import config
from dao import MySQLClient, MilvusClient
from audio.asr import PaddleSpeechRecognition, SpeechRecognitionAdapter
from audio.vector import PaddleSpeakerVerification, SpeakerVerificationAdapter, DeepSpeakerVerification
from utils.audioU import pre_process
from utils.fileU import save_file
from utils.responseU import QuickResponse as qr

# 初始化配置
ACCURACY_THRESHOLD = 0.7  # 测试用阈值
TEST_WAKE_TEXT = "四二九九八一四四六五六"  # 测试用唤醒文本

# 初始化客户端
milvus_client = MilvusClient(config.Milvus.host, config.Milvus.port)
mysql_client = MySQLClient(config.MySQL.host, config.MySQL.port, config.MySQL.user,
                          config.MySQL.password, config.MySQL.database)

# 初始化模型
paddleASR = SpeechRecognitionAdapter(PaddleSpeechRecognition())
paddleVector = SpeakerVerificationAdapter(PaddleSpeakerVerification())
deep_speakerVector = SpeakerVerificationAdapter(DeepSpeakerVerification())

def test_wake(audio_file_path, wake_text=TEST_WAKE_TEXT):
    """
    测试唤醒功能
    :param audio_file_path: 音频文件路径
    :param wake_text: 唤醒文本
    :return: 测试结果
    """
    try:
        # 保存测试音频文件
        file_path = audio_file_path
        print(f"Processing audio file: {file_path}")

        # 预处理音频文件
        pro_path = pre_process(file_path)

        # 1. 语音识别验证
        asr_result = paddleASR.recognize(file_path)
        print(f"ASR识别结果: {asr_result}")
        print(f"预期唤醒文本: {wake_text}")

        if asr_result != wake_text:
            return qr.error("唤醒文本不匹配")

        # 2. 声纹识别验证
        audio_embedding = paddleVector.get_embedding_from_file(pro_path)
        search_results = milvus_client.search(config.AUDIO_TABLE, audio_embedding, top_k=1)

        if not search_results:
            return qr.error("未找到匹配声纹")

        # 计算相似度
        similar_vector = np.array(search_results[0][0].entity.vec, dtype=np.float32)
        similarity_score = paddleVector.get_embeddings_score(similar_vector, audio_embedding)
        print(f"声纹相似度: {similarity_score}")

        if similarity_score < ACCURACY_THRESHOLD:
            return qr.error("声纹相似度不足")

        # 使用deep_speakerVector进行二次验证
        deep_speaker_embedding = deep_speakerVector.get_embedding_from_file(pro_path).squeeze()
        deep_speaker_score = deep_speakerVector.get_embeddings_score(similar_vector[np.newaxis, :], deep_speaker_embedding[np.newaxis, :])
        print(f"deep_speakerVector相似度: {deep_speaker_score}")

        if deep_speaker_score < ACCURACY_THRESHOLD:
            return qr.error("deep_speakerVector声纹相似度不足")

        # 获取用户信息
        user_id = str(search_results[0][0].id)
        user = mysql_client.get_user_by_id(user_id)
        if not user:
            return qr.error("未找到对应用户")

        # 构建成功响应
        return qr.data(
            wake_result="SUCCESS",
            user_id=user_id,
            username=user.username,
            permission_level=user.permission_level,
            similarity_score=similarity_score
        )

    except Exception as e:
        traceback.print_exc()
        return qr.error(str(e))

if __name__ == '__main__':
    # 测试用例
    test_audio_file = r"P:\xiangmu\python\Voice\Data\验证.wav"  # 替换为实际测试音频文件路径

    # 测试1: 正常情况
    print("\n测试1: 正常唤醒")
    result = test_wake(test_audio_file)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 测试2: 错误唤醒文本
    print("\n测试2: 错误唤醒文本")
    result = test_wake(test_audio_file, "错误的唤醒文本")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 测试3: 低相似度声纹 (需要准备低相似度音频)
    # print("\n测试3: 低相似度声纹")
    # low_sim_audio = "path_to_low_sim_audio.wav"
    # result = test_wake(low_sim_audio)
    # print(json.dumps(result, indent=2, ensure_ascii=False))