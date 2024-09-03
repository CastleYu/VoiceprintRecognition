import os
import zipfile


def unzip(root, file_name=None):
    if not os.path.isabs(root):
        root = os.path.abspath(root)
    if file_name is None:
        for i in os.listdir(root):
            if i.endswith('.zip'):
                unzip(root, i)
        return
    zip_file_path = os.path.join(root, file_name)

    # 打开zip文件
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # 解压所有文件到指定目录
        zip_ref.extractall(root)

    # 确保解压目录存在
    if not os.path.exists(root):
        os.makedirs(root)

    print(f"文件解压: {root}")
