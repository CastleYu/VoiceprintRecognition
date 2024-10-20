import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from action.lac_tokenizer import LACTokenizer


class InstructionMatcher:
    def __init__(self, models_dir, matcher=None):
        self.models_dir = models_dir
        self._matcher = matcher

    def match(self, input_instruction, instruction_set, threshold=0.8):
        return self._matcher.match(input_instruction, instruction_set, threshold)

    def to_vec(self, instruction):
        return self._matcher.to_vec(instruction)

    def load(self, matcher):
        self._matcher = matcher
        matcher.load_model(self.models_dir)
        return self



class MicomlMatcher:
    def __init__(self, model_name=None):
        self.model_name = model_name or 'paraphrase-multilingual-MiniLM-L12-v2'
        self._lac = LACTokenizer()  # 百度LAC分词
        self._model_bert = None
        self.threshold = 0.8
        self.cached_embeddings = None  # 缓存指令集向量
        self.cached_instructions = None  # 缓存原始指令集

    def load_model(self, model_dir):
        model_full_path = str(os.path.join(model_dir, self.model_name))
        self._model_bert = SentenceTransformer(model_full_path)
        return self

    def cache_instruction_set(self, instruction_set, file_path):
        """
        预先计算并缓存指令集的向量嵌入，并将其存储到本地文件
        """
        self.cached_instructions = instruction_set
        self.cached_embeddings = np.array(
            self._model_bert.encode([self.do_cut(instruction) for instruction in tqdm(instruction_set, desc="编码中")])
        )
        print("Step: 存储到本地文件")
        np.save(file_path, self.cached_embeddings)
        with open(file_path + '_instructions.json', 'w', encoding='utf-8') as f:
            json.dump(instruction_set, f, ensure_ascii=False)

    def load_cached_instruction_set(self, file_path):
        """
        从本地文件加载指令集向量和原始指令
        """
        self.cached_embeddings = np.load(file_path)
        with open(file_path + '_instructions.json', 'r', encoding='utf-8') as f:
            self.cached_instructions = json.load(f)

    def batch_match(self, instruction_list, threshold=None, do_cut=True):
        if self.cached_embeddings is None or self.cached_instructions is None:
            raise ValueError("请先调用 `cache_instruction_set` 或 `load_cached_instruction_set` 方法来缓存指令集。")

        threshold = threshold or self.threshold

        print("Step: 批量进行分词和编码")
        if do_cut:
            instruction_list = [self.do_cut(instr) for instr in instruction_list]
        input_embeddings = np.array(self._model_bert.encode(instruction_list))

        print("Step: 批量计算余弦相似度")
        cosine_scores = cosine_similarity(input_embeddings, self.cached_embeddings)

        print("Step: 对每个输入找到相似度最高的指令及其分数")
        results = []
        for i, scores in tqdm(enumerate(cosine_scores),desc="批量获取最高指令分数"):
            max_index = np.argmax(scores)
            max_score = scores[max_index]
            if max_score > threshold:
                results.append((self.cached_instructions[max_index], max_score))
            else:
                results.append(('No matching actions found', '0'))
        return results

    def do_cut(self, input_instruction):
        i_list = self._lac.segment(input_instruction)
        input_instruction = ''.join([i[0] for i in i_list])
        return input_instruction


    def match(self, input_instruction, instruction_set, threshold=None, do_cut=True):
        if do_cut:
            input_instruction = self.do_cut(input_instruction)
        threshold = threshold or self.threshold
        embedding = self._model_bert.encode([input_instruction])
        similarity_scores = []
        for instruction_rec in instruction_set:
            rec_embedding = self._model_bert.encode([instruction_rec])
            cosine_scores = cosine_similarity(embedding, rec_embedding)
            similarity_scores.append((instruction_rec, cosine_scores[0][0]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        if similarity_scores[0][1] > threshold:
            return similarity_scores[0][0], similarity_scores[0][1]
        else:
            return 'No matching actions found', '0'


class GoogleMatcher:
    def __init__(self, model_name=None):
        self._model = None
        self._tokenizer = None
        self.model_name = model_name or 'google-bert-base-chinese'
        self.threshold = 0.8

    def load_model(self, model_dir):
        model_full_path = str(os.path.join(model_dir, self.model_name))
        self._tokenizer = AutoTokenizer.from_pretrained(model_full_path)
        self._model = AutoModel.from_pretrained(model_full_path)

    def match(self, input_instruction, instruction_set, threshold=None):
        threshold = threshold or self.threshold
        keyword_emb = self.get_embedding(input_instruction)
        similarities = [(action, cosine_similarity(keyword_emb, self.get_embedding(action))[0][0]) for action in
                        instruction_set]

        print("Step: 按相似度排序")
        similarities.sort(key=lambda x: x[1], reverse=True)
        print(similarities)
        print("Step: 阈值检查")
        if not similarities or similarities[0][1] < threshold:
            return 'No matching actions found', '0'

        print("Step: 返回最匹配的指令")
        return similarities[0][0], similarities[0][1]

    def get_embedding(self, text):
        inputs = self._tokenizer(text, return_tensors='pt')
        outputs = self._model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()
