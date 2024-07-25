import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from action.lac_tokenizer import LACTokenizer


class InstructionMatcher:
    def __init__(self, models_dir, matcher=None):
        self.models_dir = models_dir
        self._matcher = matcher

    def match(self, input_instruction, instruction_set, threshold=0.85):
        return self._matcher.match(input_instruction, instruction_set, threshold)

    def load(self, matcher):
        self._matcher = matcher
        matcher.load_model(self.models_dir)
        return self


class MicomlMatcher:
    def __init__(self, model_name=None):
        self.model_name = model_name or 'google-bert-base-chinese'
        self._lac = None
        self._model_bert = None
        self._lac = LACTokenizer()
        self.threshold = 0.85

    def load_model(self, model_dir):
        model_full_path = str(os.path.join(model_dir, self.model_name))
        self._model_bert = SentenceTransformer(model_full_path)

    def match(self, input_instruction, instruction_set, threshold=None, do_cut=True):
        if do_cut:
            input_instruction = self.do_cut(input_instruction)
        threshold = threshold or self.threshold
        embedding = self._model_bert.encode([input_instruction])
        similarity_scores = []
        for instruction_rec in tqdm(instruction_set, desc="查找指令"):
            rec_embedding = self._model_bert.encode([instruction_rec])
            cosine_scores = cosine_similarity(embedding, rec_embedding)
            similarity_scores.append((instruction_rec, cosine_scores[0][0]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        print(similarity_scores)
        if similarity_scores[0][1] > threshold:
            return similarity_scores[0][0], similarity_scores[0][1]
        else:
            return 'No matching actions found', '0'

    def do_cut(self, input_instruction):
        i_list = self._lac.segment(input_instruction)
        input_instruction = ''.join([i[0] for i in i_list])
        return input_instruction


class GoogleMatcher:
    def __init__(self, model_name=None):
        self._model = None
        self._tokenizer = None
        self.model_name = model_name or 'paraphrase-multilingual-MiniLM-L12-v2'
        self.threshold = 0.85

    def load_model(self, model_dir):
        model_full_path = str(os.path.join(model_dir, self.model_name))
        self._tokenizer = AutoTokenizer.from_pretrained(model_full_path)
        self._model = AutoModel.from_pretrained(model_full_path)

    def match(self, input_instruction, instruction_set, threshold=None):
        threshold = threshold or self.threshold
        keyword_emb = self.get_embedding(input_instruction)
        similarities = [(action, cosine_similarity(keyword_emb, self.get_embedding(action))[0][0]) for action in
                        instruction_set]

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        print(similarities)
        # 阈值检查
        if not similarities or similarities[0][1] < threshold:
            return 'No matching actions found', '0'

        # 返回最匹配的指令
        return similarities[0][0], similarities[0][1]

    def get_embedding(self, text):
        inputs = self._tokenizer(text, return_tensors='pt')
        outputs = self._model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()


def test():
    print("start to load models")
    model_dir = os.path.join('bert_models')
    im = InstructionMatcher(model_dir)
    micoml_matcher = im.load(MicomlMatcher('paraphrase-multilingual-MiniLM-L12-v2'))
    google_matcher = im.load(GoogleMatcher('google-bert-base-chinese'))
    print("loading finished")

    instruction_set = ["打开门", "关闭门", "打开报表"]
    input_instruction = "把门开开"

    # 使用 MicomlMatcher
    result_micoml = micoml_matcher.match(input_instruction, instruction_set)
    print(f"MicomlMatcher Result: {result_micoml}")

    # 使用 GoogleMatcher
    result_google = google_matcher.match(input_instruction, instruction_set)
    print(f"GoogleMatcher Result: {result_google}")


if __name__ == '__main__':
    test()
