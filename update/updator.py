import os
import traceback

import config

RESOURCES_DIR = config.Update.ModelDir
API = config.Update.API
RESOURCES_KEYWORD = config.Update.RESOURCES_KEYWORD
MODEL_BIN_MAP = config.Model.ModelBinMap
wanted_resources = []


def get_system_proxies():
    from urllib.request import getproxies
    proxies = getproxies()
    proxies['https'] = proxies['http']
    proxies.pop('ftp')
    return proxies


def update_resources():
    if check_resources():
        print('Everything is up to date')
        return
    import requests
    response = requests.get(API, proxies=get_system_proxies())
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError("更新内容未获取到列表")

    for i in data:
        tag_name = i['tag_name']
        if RESOURCES_KEYWORD in tag_name:
            download_resources(i['assets'])
            unzip()
            check_resources() and print('Everything is up to date')
            return True
    raise ValueError("获取了更新列表，但未找到资源包")


def check_resources():
    for k, v in MODEL_BIN_MAP.items():
        path = os.path.join(config.Model.ModelDir, k, *v)
        if not os.path.exists(path):
            wanted_resources.append(k + '.zip')
    return not wanted_resources  # empty is True -> no needs update


def download_resources(assets_list):
    import requests
    from tqdm import tqdm
    for asset in assets_list:
        if asset['name'] in wanted_resources:
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
                        desc=asset['name'],
                        total=total_size_in_bytes,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(block_size):
                        f.write(data)
                        bar.update(len(data))

                print(f"\n下载资源 {asset['name']} 成功.")
            except Exception as e:
                traceback.print_exc()
                file_name = os.path.join(RESOURCES_DIR, asset['name'])
                print(f"下载资源 {asset['name']} 失败: {type(e).__name__}")
                if os.path.exists(file_name):
                    os.remove(file_name)

def unzip(file_name=None):
    import zipfile
    root = RESOURCES_DIR
    if not os.path.isabs(root):
        root = os.path.abspath(root)
    if file_name is None:
        for i in os.listdir(root):
            if i.endswith('.zip'):
                unzip(i)
        return
    zip_file_path = os.path.join(root, file_name)

    # 确保解压目录存在
    if not os.path.exists(root):
        os.makedirs(root)

    # 打开zip文件
    try:
        with zipfile.ZipFile(str(zip_file_path), 'r') as zip_ref:
            # 解压所有文件到指定目录
            zip_ref.extractall(root)
        os.remove(zip_file_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"解压失败 {zip_file_path}\n\t{type(e).__name__}")
    print(f"文件解压: {root}")


if __name__ == '__main__':
    update_resources()
