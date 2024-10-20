import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

import config
from action.action_matcher import MicomlMatcher

# 读取模型目录
DIR = config.Model.ModelDir
mm = MicomlMatcher().load_model(DIR)

# 读取数据集
df = pd.read_csv("Chinese_Text_Similarity.txt", sep='\t')
instruction_set = list(df["句子2"])

# 缓存指令集到本地文件并加载
cache_file_path = "cached_embeddings.npy"
mm.cache_instruction_set(instruction_set, cache_file_path)
mm.load_cached_instruction_set(cache_file_path)

# 定义存储真实值和预测值的列表
y_true = df["相似度"].tolist()
y_pred = []
mismatch_results = []  # 用于存储不符合预期的结果

# 获取所有句子1的列表
left_sentences = df["句子1"].tolist()

# 批量预测
predictions = mm.batch_match(left_sentences)

# 遍历批量预测的结果
for idx, (matched_sentence, score) in tqdm(enumerate(predictions)):
    right = df.loc[idx, '句子2']
    similarity = df.loc[idx, '相似度']

    # 计算匹配结果是否与“句子2”一致
    is_match = (matched_sentence == right)
    predicted_similarity = 1.0 if is_match else 0.0

    # 保存预测值
    y_pred.append(predicted_similarity)

    # 如果预测结果与期望不符，记录详细信息
    if predicted_similarity != similarity:
        mismatch_results.append({
            "res": matched_sentence,
            "left": left_sentences[idx],
            "expected_ans": right,
            "expected_similarity": similarity,
            "predicted_similarity": predicted_similarity
        })

# 计算指标
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

# 打印不符合预期的结果
if mismatch_results:
    print("\nMismatched Results:")
    for idx, mismatch in enumerate(mismatch_results, 1):
        print(f"\nCase {idx}:")
        print(f"Left sentence: {mismatch['left']}")
        print(f"Predicted Sentence: {mismatch['res']}")
        print(f"Expected Sentence: {mismatch['expected_ans']}")
        print(f"Expected Similarity: {mismatch['expected_similarity']}")
        print(f"Predicted Similarity: {mismatch['predicted_similarity']}")
else:
    print("All cases passed!")

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
