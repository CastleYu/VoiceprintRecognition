from abc import ABC, abstractmethod
import paddle
from paddlespeech.cli.asr import ASRExecutor

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
    def recognize(self, audio_file, lang, sample_rate):
        # 调用ASRExecutor进行语音识别
        asr_executor = ASRExecutor()
        text = asr_executor(
            model='conformer_wenetspeech',
            lang=lang,
            sample_rate=sample_rate,
            config=None,
            ckpt_path=None,
            audio_file=audio_file,
            force_yes=False,
            device=paddle.get_device())
        return text
