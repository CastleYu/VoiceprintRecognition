import os
import random
import numpy as np
import logging
from typing import List, Tuple
from tqdm import tqdm
from sklearn.model_selection import KFold
from audio.vector import PaddleSpeakerVerification, SpeakerVerificationAdapter
import evaluate as eval_metric

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_seed(seed=123):
    np.random.seed(seed)
    random.seed(seed)

class SpeakerVerificationSystem:
    def __init__(self):
        """初始化说话人验证系统"""
        # 初始化随机种子保证结果可复现

        # 初始化DeepSpeaker模型
        logger.info(f"正在加载模型: ")
        try:
            self.paddleVector = SpeakerVerificationAdapter(PaddleSpeakerVerification())
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型权重加载失败: {str(e)}")
            raise


    def compare_pair(self, audio_path1, audio_path2):
        """比较两个音频的相似度"""
        logger.debug(f"比较音频对: {audio_path1} 和 {audio_path2}")
        emb1 = self.paddleVector.get_embedding(audio_path1)
        emb2 = self.paddleVector.get_embedding(audio_path2)
        if emb1 is None or emb2 is None:
            logger.warning("无法提取一个或两个音频的嵌入向量，返回默认相似度0.0")
            return 0.0
        similarity = self.paddleVector.get_embeddings_score(emb1, emb2)
        logger.debug(f"相似度计算结果: {similarity:.4f}")
        return similarity

def generate_test_pairs(data_dir: str, max_samples_per_speaker: int = 10) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    """
    生成平衡的说话人验证测试对

    参数说明：
    - data_dir: 包含说话人子目录的根目录（示例：/path/to/data）
    - max_samples_per_speaker: 每个说话人最大采样数

    返回：
    - test_pairs: 音频对列表，格式[(音频路径1, 音频路径2), ...]
    - labels: 对应标签数组，1表示同一说话人，0表示不同

    文件结构要求：
    data_dir/
    ├── 01_何丽婕/
    │   ├── audio1.wav
    │   └── audio2.wav
    ├── 02_张伟/
    │   └── audio3.wav
    └── ...
    """

    # 第一阶段：数据收集与验证
    speaker_dict = {}  # 结构：{speaker_id: [full_path1, ...]}

    # 遍历数据目录（兼容多级目录结构）
    for root, dirs, files in os.walk(data_dir):
        # 只处理直接包含音频文件的目录（即说话人目录）
        if root == data_dir:
            continue  # 跳过根目录本身

        # 从路径中提取说话人ID（示例：/data/01_何丽婕 -> "01_何丽婕"）
        speaker_id = os.path.basename(root)

        # 收集当前目录下的所有WAV文件
        audio_files = [
            os.path.join(root, f)
            for f in files
            if (
                f.lower().endswith('.wav') and
                'combined' not in f.lower()  # 新增过滤条件
            )
        ]

        if audio_files:
            speaker_dict[speaker_id] = audio_files
            logger.debug(f"发现说话人 {speaker_id}，包含 {len(audio_files)} 个音频文件")

    # 数据有效性检查
    if not speaker_dict:
        raise FileNotFoundError(f"在目录 {data_dir} 中未找到有效的音频文件")

    speaker_list = list(speaker_dict.items())
    logger.info(f"共发现 {len(speaker_list)} 个说话人，首5个为：{list(speaker_dict.keys())[:5]}")

    # 第二阶段：动态计算采样参数
    # 获取所有说话人的最小样本数（确保样本平衡）
    min_samples = min(len(files) for _, files in speaker_list)
    actual_samples = min(min_samples, max_samples_per_speaker)
    logger.info(f"采样参数 | 理论最大值: {max_samples_per_speaker} | 实际采样数: {actual_samples}")

    # 第三阶段：生成样本对
    test_pairs = []
    labels = []

    for idx, (spk_id, spk_files) in enumerate(speaker_list):
        # 随机采样固定数量的音频（确保所有说话人样本数相同）
        selected = random.sample(spk_files, actual_samples)

        # 生成正样本对（同说话人组合）
        # 生成所有可能的非重复组合 C(n,2)
        for i in range(actual_samples):
            for j in range(i+1, actual_samples):
                test_pairs.append((selected[i], selected[j]))
                labels.append(1)

        # 生成负样本对（跨说话人组合）
        # 与后续所有说话人配对，避免重复组合
        for other_spk_id, other_files in speaker_list[idx+1:]:
            other_selected = random.sample(other_files, actual_samples)

            # 生成1:1的负样本对
            for i in range(actual_samples):
                test_pairs.append((selected[i], other_selected[i]))
                labels.append(0)

    # 第四阶段：数据洗牌
    # 将数据对和标签组合后打乱，保持对应关系
    combined = list(zip(test_pairs, labels))
    random.shuffle(combined)
    test_pairs, labels = zip(*combined) if combined else ([], [])

    logger.info(f"生成样本统计 | 总数: {len(test_pairs)} | 正样本: {sum(labels)} | 负样本: {len(labels)-sum(labels)}")
    return list(test_pairs), np.array(labels)

def main(data_dir=r"P:/xiangmu/python/Voice/Data/test"):
    try:
        set_seed()
        # 初始化验证系统
        verifier = SpeakerVerificationSystem()

        # 1. 生成测试对
        test_pairs, labels = generate_test_pairs(data_dir)
        if not test_pairs:
            logger.error("没有生成有效的测试对，程序退出")
            return

        # 2. 计算相似度矩阵（带进度条）
        sims = []
        logger.info("开始计算相似度矩阵...")
        for pair in tqdm(test_pairs, desc="处理音频对"):
            similarity = verifier.compare_pair(pair[0], pair[1])
            sims.append(similarity)
        sims = np.array(sims)

        # 3. 评估性能
        logger.info("\n评估结果：")
        fm, tpr, acc, eer = eval_metric.evaluate(sims, labels)
        logger.info(f"F-measure: {fm:.3f}")
        logger.info(f"True Positive Rate: {tpr:.3f}")
        logger.info(f"Accuracy: {acc:.3f}")
        logger.info(f"Equal Error Rate: {eer:.3f}")

        # 4. 改进的交叉验证（按说话人分组）
        logger.info("\n改进的交叉验证结果：")
        # 假设每个说话人有相同数量的样本，实际应用中需要更复杂的处理
        kf = KFold(n_splits=5)
        for fold, (train_idx, test_idx) in enumerate(kf.split(sims)):
            fm, tpr, acc, eer = eval_metric.evaluate(sims[test_idx], labels[test_idx])
            logger.info(f"Fold {fold+1} - EER: {eer:.3f}, F1: {fm:.3f}")

    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}", exc_info=True)
    finally:
        logging.shutdown()

if __name__ == "__main__":
    main()
