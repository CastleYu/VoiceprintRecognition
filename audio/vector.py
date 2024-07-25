from abc import ABC, abstractmethod

import paddle
from paddlespeech.cli.vector import VectorExecutor


class SpeakerVerification(ABC):
    @abstractmethod
    def get_embedding(self, audio_file, sample_rate):
        pass

    @abstractmethod
    def get_embeddings_score(self, emb1, emb2):
        pass


# 适配器
class SpeakerVerificationAdapter(SpeakerVerification):
    def __init__(self, adaptee):
        self.adaptee = adaptee

    def get_embedding(self, audio_file, sample_rate=16000):
        return self.adaptee.get_embedding(audio_file, sample_rate)

    def get_embeddings_score(self, emb1, emb2):
        return self.adaptee.get_embeddings_score(emb1, emb2)


# PaddleSpeech实现类
class PaddleSpeakerVerification:
    def __init__(self):
        self.vector_executor = VectorExecutor()

    def get_embedding(self, audio_file, sample_rate):
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
