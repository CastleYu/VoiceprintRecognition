import requests
from requests.compat import getproxies as get_system_proxies
from tqdm import tqdm

API = 'https://gitee.com/api/v5/repos/muziyuc/VoiceprintRecognition/releases?page=1&per_page=20'
WANTED_RESOURCES = ['google-bert-base-chinese.zip', 'paraphrase-multilingual-MiniLM-L12-v2.zip']


def get_resources():
    response = requests.get(API, proxies=get_system_proxies())
    data = response.json()
    if not isinstance(data, list):
        raise ValueError("更新内容未获取到列表")

    for i in data:
        tag_name = i['tag_name']
        if 'resources' in tag_name:
            download_resources(i['assets'])
            return True

    raise ValueError("获取了更新列表，但未找到资源包")


def download_resources(assets_list):
    for asset in assets_list:
        if asset['name'] in WANTED_RESOURCES:
            try:
                # 启用流式传输
                response = requests.get(asset['browser_download_url'], stream=True, proxies=get_system_proxies())
                response.raise_for_status()  # 检查请求是否成功

                # 获取文件大小信息
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024  # 每块的大小
                file_name = asset['name']

                # 使用 tqdm 显示下载进度条
                with open(file_name, 'wb') as f, tqdm(
                        desc=file_name,
                        total=total_size_in_bytes,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(block_size):
                        f.write(data)
                        bar.update(len(data))

                print(f"\nDownloaded {file_name} successfully.")
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {asset['name']}: {e}")
