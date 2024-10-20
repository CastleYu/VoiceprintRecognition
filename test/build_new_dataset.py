# from action.action_matcher import MicomlMatcher, GoogleMatcher
from data_set import *
import pandas as pd
import random

# mm = MicomlMatcher()
# gm = GoogleMatcher()
data = read_csv_like("Chinese_Text_Similarity.txt", sep='\t', has_header=True)
cts = DataSet(data)
df = cts.df
from tqdm import tqdm

# 统计正负样本
positive_samples = cts.where(相似度=lambda x: eval(x) > 0).df
negative_samples = cts.where(相似度=lambda x: eval(x) == 0).df

num_positive = len(positive_samples)
num_negative = len(negative_samples)

# 计算所需的样本数量（根据比例）
# 假设我们想生成 20 个样本，比例根据正负样本数量动态调整
total_samples = len(df)
ratio = num_positive / (num_positive + num_negative)
num_required_positive = int(total_samples * ratio)
num_required_negative = total_samples - num_required_positive

# 新数据集构造
new_data = []

# 随机选择正样本
for _ in tqdm(range(num_required_positive)):
    pos_row = positive_samples.sample(n=1).iloc[0]
    left_sentence = pos_row['句子1']
    produced_right_sentence = pos_row['句子2']

    candidates = list(set(df["句子2"]))

    candidate_right_sentences = random.sample(
        candidates,
        k=random.randint(min(3, len(df['句子2']) - 1), 40)
    )
    candidate_right_sentences.append(produced_right_sentence)
    random.shuffle(candidate_right_sentences)

    new_data.append({
        '左句': left_sentence,
        '候选右句集合': candidate_right_sentences,
        '产生右句': produced_right_sentence,
        '关联': 1  # 正样本
    })

# 随机选择负样本
for _ in tqdm(range(num_required_negative)):
    neg_row = negative_samples.sample(n=1).iloc[0]
    left_sentence = neg_row['句子1']
    produced_right_sentence = neg_row['句子2']

    candidates = list(set(df["句子2"]))
    candidate_right_sentences = random.sample(
        candidates,
        k=random.randint(min(3, len(df['句子2']) - 1), 10)
    )
    candidate_right_sentences.append(produced_right_sentence)
    random.shuffle(candidate_right_sentences)

    new_data.append({
        '左句': left_sentence,
        '候选右句集合': candidate_right_sentences,
        '产生右句': produced_right_sentence,
        '关联': 0  # 负样本
    })

# 创建新的 DataFrame
random.shuffle(new_data)
new_df = pd.DataFrame(new_data)
instruction_judge = DataSet(new_df).save_csv('InstructionSet_fix.csv')
