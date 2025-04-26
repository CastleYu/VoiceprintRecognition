import re
import csv

def extract_matching_logs(log_file_path, output_csv_path):
    """
    Extract logs where speaker and user info match, and save to CSV.
    CSV columns: speaker, similarity, file.
    """
    pattern = re.compile(
        r'说话人 (speakerF\d{4}) 的声纹向量查询完成，相似度: ([0-9.]+), 用户信息: (speakerF\d{4}), 音频文件为：(.*)'
    )
    
    matching_logs = []
    
    with open(log_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            match = pattern.search(line)
            if match:
                speaker, similarity, user_info, file_path = match.groups()
                if speaker == user_info:
                    matching_logs.append([speaker, similarity, file_path])
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['speaker', 'similarity', 'file'])
        writer.writerows(matching_logs)

if __name__ == "__main__":
    log_file_path = "test_backup.log"
    output_csv_path = "speaker_records.csv"
    extract_matching_logs(log_file_path, output_csv_path)