import os
import json
from test_audio_recognition import test_audio_recognition

def run_test_case(audio_file, description):
    print(f"\n=== 测试用例: {description} ===")
    print(f"音频文件: {audio_file}")
    result = test_audio_recognition(audio_file)
    print("测试结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

if __name__ == '__main__':
    # 测试用例1: 仅使用Paddle模型（长文本）
    long_text_audio = r"P:\xiangmu\python\Voice\Data\test\21_蔡培\combined_20250323_225137.wav"  # 替换为实际长文本音频路径
    run_test_case(long_text_audio, "长文本音频（仅使用Paddle模型）")

    # 测试用例2: 仅使用DeepSpeaker模型（短文本）
    short_text_audio =  r"P:\xiangmu\python\Voice\Data\test\21_蔡培\[2025-03-03][18-41-22].wav" # 替换为实际短文本音频路径
    run_test_case(short_text_audio, "短文本音频（仅使用DeepSpeaker模型）")

    # 测试用例3: 同时使用两种模型（中等长度文本）
    medium_text_audio = r"P:\xiangmu\python\Voice\Data\test\21_蔡培\[2025-03-03][18-41-50].wav"  # 替换为实际中等长度文本音频路径
    run_test_case(medium_text_audio, "中等长度文本音频（同时使用两种模型）")

    print("\n所有测试用例执行完成")