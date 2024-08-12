from abc import ABC, abstractmethod

import paddle
from paddlespeech.cli.asr import ASRExecutor

version = paddle.__version__
if version != '2.4.1':
    raise RuntimeError(f"Paddle版本不正确，期望为2.4.1，实际为{version}")


# from paddlespeech.cli.text import TextExecutor

class SpeechRecognition(ABC):
    @abstractmethod
    def recognize(self, audio_file, lang, sample_rate):
        pass


# 适配器
class SpeechRecognitionAdapter(SpeechRecognition):
    def __init__(self, adaptee):
        self.adaptee = adaptee

    def recognize(self, audio_file, lang='zh', sample_rate=16000):
        return self.adaptee.recognize(audio_file, lang, sample_rate)


# PaddleSpeech实现类
class PaddleSpeechRecognition:
    def __init__(self):
        self.asr_executor = ASRExecutor()
        # self.text_executor = TextExecutor()

    def recognize(self, audio_file, lang, sample_rate):
        # 调用ASRExecutor进行语音识别
        text = self.asr_executor(
            model='conformer_wenetspeech',
            lang=lang,
            sample_rate=sample_rate,
            config=None,
            ckpt_path=None,
            audio_file=audio_file,
            force_yes=True,
            device=paddle.get_device())
        # text = self.text_executor(text=text)
        return text
