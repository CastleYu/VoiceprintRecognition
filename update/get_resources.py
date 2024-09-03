import os.path
import traceback

import requests
from requests.compat import getproxies
from tqdm import tqdm

from config import ROOT_DIR

API = 'https://api.github.com/repos/CastleYu/VoiceprintRecognition/releases'
RESOURCES_KEYWORD = 'resources'
WANTED_RESOURCES = ['google-bert-base-chinese.zip', 'paraphrase-multilingual-MiniLM-L12-v2.zip']
RESOURCES_DIR = os.path.join(ROOT_DIR, 'action', 'bert_models')


def get_system_proxies():
    proxies = getproxies()
    proxies['https'] = proxies['http']
    proxies.pop('ftp')
    return proxies


def update_resources():
    response = requests.get(API, proxies=get_system_proxies())
    data = response.json()
    if not isinstance(data, list):
        raise ValueError("更新内容未获取到列表")

    for i in data:
        tag_name = i['tag_name']
        if RESOURCES_KEYWORD in tag_name:
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
                file_name = os.path.join(RESOURCES_DIR, asset['name'])

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

                print(f"\n下载资源 {asset['name']} 成功.")
            except requests.exceptions.RequestException as e:
                traceback.print_exc()
                file_name = os.path.join(RESOURCES_DIR, asset['name'])
                print(f"下载资源 {asset['name']} 失败: {e}")
                if os.path.exists(file_name):
                    os.remove(file_name)


if __name__ == '__main__':
    update_resources()
