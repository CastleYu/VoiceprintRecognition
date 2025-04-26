import numpy as np
from storage import VoiceprintStorage
from faker import Faker
import random
import os
import sys
from pathlib import Path
import librosa
import soundfile as sf
import logging
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import config

SAMPLE_RATE = 16000  # Standard sample rate for voice processing
from audio.vector import PaddleSpeakerVerification, SpeakerVerificationAdapter, DeepSpeakerVerification
from const import SUCCESS, FAILED
from dao import *
from utils.audioU import pre_process

ACCURACY_THRESHOLD = config.Algorithm.threshold
LOW_TEXT_CHARACTER_LIMIT = config.Algorithm.low_text_threshold
HIGH_TEXT_CHARACTER_LIMIT = config.Algorithm.high_text_threshold

# 输入目录路径
input_dir = r"P:\xiangmu\python\Voice\Data\testOri"
# 配置日志
input_dir_name = os.path.basename(input_dir)
log_filename = f'test_{input_dir_name}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置随机种子以确保结果可复现
np.random.seed(42)
random.seed(42)

# 初始化存储类
storage = VoiceprintStorage()

#初始化模型
paddleVector = SpeakerVerificationAdapter(PaddleSpeakerVerification())
deep_speakerVector = SpeakerVerificationAdapter(DeepSpeakerVerification())

# 初始化 Faker
fake = Faker()


# 获取所有说话人及其音频文件
speaker_audio_map = {}
for audio_file in Path(input_dir).rglob('*.wav'):
    stem = audio_file.stem
    parts = stem.split('_')

    # 尝试从父目录名中提取说话人(格式:数字_姓名)
    parent_dir = audio_file.parent.name
    if '_' in parent_dir:
        dir_parts = parent_dir.split('_')
        if len(dir_parts) == 2 and dir_parts[0].isdigit():
            speaker = dir_parts[1]
        else:
            speaker = None
    else:
        speaker = None

    # 如果无法从目录名提取，尝试从文件名提取
    if speaker is None:
        if len(parts) == 2:
            speaker = parts[0]
        elif len(parts) == 3 and parts[0] == 'speaker':
            speaker = parts[1]
        elif len(parts) >= 4 and parts[0].startswith('F'):
            speaker = parts[0]
        elif '-' in stem:
            speaker = stem.split('-')[0]
        else:
            continue

    if speaker not in speaker_audio_map:
        speaker_audio_map[speaker] = []
    speaker_audio_map[speaker].append(audio_file)
logger.info(f'成功加载 {len(speaker_audio_map)} 个说话人的音频文件')

def concatenate_audio(audio_files):
    """拼接多个音频文件为一个音频信号，并保存为临时文件"""
    from pydub import AudioSegment
    import tempfile

    combined = AudioSegment.empty()
    for file in audio_files:
        audio = AudioSegment.from_file(file)
        combined += audio

    # 生成临时文件
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    combined.export(temp_file.name, format="wav")
    temp_file.close()

    return temp_file.name

# 从每个说话人中选取音频文件进行拼接
for speaker, audio_files in speaker_audio_map.items():
    # 检查数据库是否已存在该说话人
    if storage.get_user_by_name(speaker):
        print(f'Speaker {speaker} already exists in the database. Skipping.')
        continue

    while True:
        selected_files = random.sample(audio_files, min(3, len(audio_files)))  # 最多选择3个文件
        temp_audio_path = concatenate_audio(selected_files)

        # 计算拼接后的音频时长
        duration = librosa.get_duration(filename=temp_audio_path)

        # 如果时长在5~6秒之间，则继续处理
        if 6 <= duration <= 8:
            break
        # 如果时长不足5秒，则重新选取文件
        elif duration < 6:
            os.unlink(temp_audio_path)  # 删除临时文件
            continue
        # 如果时长超过6秒，则裁剪到6秒
        else:
            signal, _ = librosa.load(temp_audio_path, sr=SAMPLE_RATE, mono=True)
            max_samples = 6 * SAMPLE_RATE
            signal = signal[:max_samples]
            sf.write(temp_audio_path, signal, SAMPLE_RATE)
            break

    # 处理音频文件
    pro_path = pre_process(temp_audio_path)
    emb_192 = np.array(paddleVector.get_embedding_from_file(pro_path), dtype=np.float32).reshape(1, -1)
    emb_512 = deep_speakerVector.get_embedding_from_file(pro_path)

    # 添加声纹向量
    index_id_189, index_id_512 = storage.add_voiceprint(emb_192, emb_512)
    logger.info(f'说话人 {speaker} 的声纹向量已添加，ID: {index_id_189}, {index_id_512}')

    # 添加用户信息
    other_info = fake.text(max_nb_chars=50)
    user_id = storage.add_user(speaker, other_info, index_id_189, index_id_512)
    logger.info(f'用户 {speaker} 已添加，用户ID: {user_id}')

    # 保留拼接音频信息
    storage.save_audio_info(speaker, [str(f) for f in selected_files])
    logger.info(f'说话人 {speaker} 的音频文件信息已保存')

    # 删除临时文件
    os.unlink(temp_audio_path)
    logger.info(f'临时文件 {temp_audio_path} 已删除')

# 全量验证
logger.info('开始全量验证...')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# 初始化统计变量
true_labels = []
predicted_labels = []
similarity_scores = []

# 新增参数：验证比例，默认为1（抽取一条音频）
verification_ratio = 1.0   # 支持设置为0.1~1.0的比例

for speaker in speaker_audio_map.keys():
    # 从`save_audio_info`中读取已用于录入的文件
    saved_files = storage.get_audio_info(speaker)
    # 从`speaker_audio_map`中读取当前说话人的所有音频文件
    all_files = speaker_audio_map[speaker]
    # 如果`saved_files`不为空，则排除已用于录入的文件
    verification_files = all_files if not saved_files else [f for f in all_files if str(f) not in saved_files]

    def process_verification_file(verification_file):
        """处理单个验证文件"""
        y, sr = librosa.load(verification_file, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        text_estimate = duration * (110 / 60)  # 取中间值110字/min

        pro_path = pre_process(verification_file)

        # 根据文本量决定模型组合
        use_paddle = text_estimate >= LOW_TEXT_CHARACTER_LIMIT
        use_deep = text_estimate <= HIGH_TEXT_CHARACTER_LIMIT

        # 获取音频嵌入向量
        paddle_embedding = paddleVector.get_embedding_from_file(pro_path) if use_paddle else None
        deep_embedding = deep_speakerVector.get_embedding_from_file(pro_path) if use_deep else None

        # 查询声纹向量
        distances_192, indices_192, vectors_192 = (storage.search_voiceprint_192(paddle_embedding)
                                          if use_paddle else (None, None, None))
        distances_512, indices_512, vectors_512 = (storage.search_voiceprint_512(deep_embedding)
                                          if use_deep else (None, None, None))

        similarity_score = 0.0
        user_info = None
        # 计算余弦相似度
        if use_paddle and use_deep:
            # 使用两种模型的情况
            paddle_match = paddleVector.get_embeddings_score(
                vectors_192.squeeze(), paddle_embedding) >= ACCURACY_THRESHOLD
            deep_match = deep_speakerVector.get_embeddings_score(
               vectors_512,deep_embedding)[0] >= ACCURACY_THRESHOLD

            recognize_result = SUCCESS if paddle_match or deep_match else FAILED

            if recognize_result == SUCCESS:
                user_info = storage.get_user_by_index_id_192(int(indices_192[0][0])) if paddle_match else storage.get_user_by_index_id_512(int(indices_512[0][0]))
                similarity_score = max(
                    paddleVector.get_embeddings_score(
                        vectors_192.squeeze(), paddle_embedding) if paddle_match else 0,
                    deep_speakerVector.get_embeddings_score(
                        vectors_512, deep_embedding)[0] if deep_match else 0
                )
        elif use_paddle:
            # 仅使用paddle模型
            similarity_score = paddleVector.get_embeddings_score(
               vectors_192.squeeze(), paddle_embedding)
            recognize_result = SUCCESS if similarity_score >= ACCURACY_THRESHOLD else FAILED
            user_info = storage.get_user_by_index_id_192(int(indices_192[0][0]))
        elif use_deep:
            # 仅使用deep模型
            similarity_score = deep_speakerVector.get_embeddings_score(
                vectors_512, deep_embedding)[0]
            recognize_result = SUCCESS if similarity_score>= ACCURACY_THRESHOLD else FAILED
            user_info = storage.get_user_by_index_id_512(int(indices_512[0][0]))

        logger.info(f'说话人 {speaker} 的声纹向量查询完成，相似度: {similarity_score:.4f}, 用户信息: {user_info[1]}, 音频文件为：{verification_file}')
        return similarity_score, user_info

    # 按比例抽取验证文件
    num_files_to_verify = max(1, int(len(verification_files) * verification_ratio))
    selected_files = random.sample(verification_files, num_files_to_verify)
    logger.info(f'验证说话人 {speaker}，使用文件: {selected_files}')

    for verification_file in selected_files:
        similarity_score, user_info = process_verification_file(verification_file)
        similarity_scores.append(similarity_score)
        true_labels.append(speaker)
        predicted_labels.append(user_info[1])

# 计算评价指标
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')
logger.info(f'评价指标计算完成 - 准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}')

# 计算FAR和ERR
thresholds = np.linspace(0, 1, 100)
far_list = []
frr_list = []

for threshold in thresholds:
    far = 0
    frr = 0
    total_imposter = 0
    total_genuine = 0

    for i in range(len(true_labels)):
        if true_labels[i] != predicted_labels[i]:
            total_imposter += 1
            if similarity_scores[i] >= threshold:
                far += 1
        else:
            total_genuine += 1
            if similarity_scores[i] < threshold:
                frr += 1

    if total_imposter > 0:
        far_list.append(far / total_imposter)
    else:
        far_list.append(0)

    if total_genuine > 0:
        frr_list.append(frr / total_genuine)
    else:
        frr_list.append(0)

# 计算ERR
err_idx = np.argmin(np.abs(np.array(far_list) - np.array(frr_list)))
err_threshold = thresholds[err_idx]
err = (far_list[err_idx] + frr_list[err_idx]) / 2
logger.info(f'生物识别指标计算完成 - FAR: {far_list[err_idx]:.4f}, FRR: {frr_list[err_idx]:.4f}, ERR: {err:.4f} (阈值: {err_threshold:.4f})')

# 关闭连接
storage.close()
logger.info('数据库连接已关闭')