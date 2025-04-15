import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import csv
import time
import traceback
import librosa
from audio.asr import PaddleSpeechRecognition, SpeechRecognitionAdapter

paddleASR = SpeechRecognitionAdapter(PaddleSpeechRecognition())

def batch_asr_test_single(input_folder, output_csv, file_extensions=['.wav', '.mp3']):
    """
    单线程版批量ASR测试函数
    :param input_folder: 音频文件夹路径
    :param output_csv: 结果文件路径
    :param file_extensions: 支持的音频格式
    """

    # 收集音频文件
    audio_files = []
    audio_files.append(r'P:\xiangmu\python\Voice\opendoor.wav')
    # for root, _, files in os.walk(input_folder):
    #     for f in files:
    #         if any(f.lower().endswith(ext) for ext in file_extensions):
    #             audio_files.append(os.path.join(root, f))

    if not audio_files:
        print(f"在 {input_folder} 中未找到支持的音频文件")
        return

    # 准备结果文件
    total_files = len(audio_files)
    success_count = 0
    start_time = time.time()

    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(['文件路径', '音频时长(s)', '识别文本', '处理时间(s)', '状态'])

        for idx, file_path in enumerate(audio_files, 1):
            file_start = time.time()
            duration = 0.0
            text = ''
            status = '成功'

            try:
                # 获取音频时长
                duration = librosa.get_duration(path=file_path)

                # 语音识别
                text = paddleASR.recognize(audio_file=file_path)
                success_count += 1

                # 实时进度显示
                print(f"处理中 [{idx}/{total_files}]")
                print(f"文件: {os.path.basename(file_path)}")
                print(f"时长: {duration:.1f}s → 结果: {text}")

            except Exception as e:
                status = f'失败: {str(e)}'
                traceback.print_exc()

            finally:
                processing_time = time.time() - file_start
                writer.writerow([
                    file_path,
                    f"{duration:.2f}",
                    text,
                    f"{processing_time:.2f}",
                    status
                ])

    # 统计信息
    total_time = time.time() - start_time
    total_duration = sum(librosa.get_duration(path=f) for f in audio_files)

    print(f"\n测试报告:")
    print(f"总处理文件: {total_files}")
    print(f"成功识别率: {success_count}/{total_files} ({success_count/total_files:.1%})")
    print(f"总音频时长: {total_duration:.1f}秒")
    print(f"总耗时: {total_time:.1f}秒")
    print(f"实时率: {total_time/total_duration:.1f}x")

# 使用示例
if __name__ == "__main__":
    batch_asr_test_single(
        input_folder=r'P:\xiangmu\python\Voice\Data\A组(1)\A组',
        output_csv=r'P:\xiangmu\python\Voice\asr_single_results.csv'
    )
