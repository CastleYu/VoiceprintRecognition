# -*- coding:utf-8 -*-
import json
import time
from action.joint.detector import JointIntentSlotDetector


class IntentRecognition:
    def __init__(self, model_path, intent_label_path, slot_label_path):
        self.model_path = model_path
        self.tokenizer_path = model_path
        self.intent_label_path = intent_label_path
        self.slot_label_path = slot_label_path
        self.model = None
        self.load_time = None
        self._initialize_model()

    def _initialize_model(self):
        start_time = time.perf_counter()
        self.model = JointIntentSlotDetector.from_pretrained(
            model_path=self.model_path,
            tokenizer_path=self.tokenizer_path,
            intent_label_path=self.intent_label_path,
            slot_label_path=self.slot_label_path
        )
        self.load_time = time.perf_counter() - start_time
        # print(f"模型加载完成，加载时间：{self.load_time:.4f}秒")

    def detect_intent(self, text):
        """检测输入文本的意图和槽位"""
        if not self.model:
            self._initialize_model()
        return self.model.detect(text.strip())

