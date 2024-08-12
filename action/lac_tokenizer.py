import os

from LAC import LAC


# 百度LAC分词器，带<去停用词>功能
class LACTokenizer:
    def __init__(self, seg_mode='lac', stopwords_file='stopwords.txt'):
        if stopwords_file:
            self.load_stopwords(stopwords_file)
        else:
            self._stopwords = []
        self._lac = LAC(mode=seg_mode)

    def load_stopwords(self, filepath):
        if not os.path.exists(filepath):
            filepath = os.path.join((os.path.dirname(os.path.abspath(__file__))), 'stopwords.txt')
        with open(filepath, 'r', encoding='utf-8') as f:
            stopwords = f.read().splitlines()
        self._stopwords = set(stopwords)

    def segment(self, text):
        lac_result = self._lac.run(text)
        words = list(zip(*lac_result))
        words = [word for word in words if word[0] not in self._stopwords and word[0].strip()]
        return words
