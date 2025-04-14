import random
import sys
import numpy as np
import paddle
from paddlespeech.cli.vector import VectorExecutor
from PyQt5 import QtWidgets, QtCore

import config
from audio.deep_speaker.audio import read_mfcc
from audio.deep_speaker.batcher import sample_from_mfcc
from audio.deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from audio.deep_speaker.conv_models import DeepSpeakerModel
from audio.deep_speaker.test import batch_cosine_similarity

version = paddle.__version__
if version != '2.4.1':
    raise RuntimeError(f"Paddle版本不正确，期望为2.4.1，实际为{version}")


class SpeakerVerificationAdapter:
    def __init__(self, adaptee):
        self.verificator = adaptee

    def get_embedding_from_file(self, audio_file, sample_rate=16000):
        return self.verificator.get_embedding_from_file(audio_file, sample_rate)

    def get_embeddings_score(self, emb1, emb2):
        return self.verificator.get_embeddings_score(emb1, emb2)


# PaddleSpeech实现类
class PaddleSpeakerVerification:
    def __init__(self):
        self.vector_executor = VectorExecutor()

    def get_embedding_from_file(self, audio_file, sample_rate):
        audio_emb = self.vector_executor(
            model='ecapatdnn_voxceleb12',
            sample_rate=sample_rate,
            config=None,
            ckpt_path=None,
            audio_file=audio_file,
            force_yes=True,
            device=paddle.get_device())
        return audio_emb

    def get_embeddings_score(self, emb1, emb2):
        return self.vector_executor.get_embeddings_score(emb1, emb2)


# DeepSpeaker实现类：封装原有示例逻辑
class DeepSpeakerVerification:
    def __init__(self, weight_path=config.Model.DeepSpeakerModelDir):
        # 为了结果可重复，每次使用相同随机数种子
        np.random.seed(123)
        random.seed(123)

        # 初始化DeepSpeaker模型并加载权重
        self.model = DeepSpeakerModel()
        self.model.m.load_weights(weight_path, by_name=True)

    def get_embedding_from_file(self, audio_file, sample_rate=SAMPLE_RATE):
        """
        从指定音频文件中获取语音嵌入，参数sample_rate默认为deep_speaker中的SAMPLE_RATE。
        """
        # 读取音频文件的mfcc特征，并截取固定帧数
        mfcc = sample_from_mfcc(read_mfcc(audio_file, sample_rate), NUM_FRAMES)
        # 通过模型预测得到嵌入，shape为(1, 512)
        embedding = self.model.m.predict(np.expand_dims(mfcc, axis=0))
        return embedding

    def get_embeddings_score(self, emb1, emb2):
        """
        计算两个嵌入向量间的余弦相似度。
        """
        return batch_cosine_similarity(emb1, emb2)
