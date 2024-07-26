# 声纹/指令识别

https://github.com/tttungwu/Speech

## build

### 创建conda环境

```sh
conda create --name speech python=3.8
```

### 安装必要的包

以下几组选一组运行即可，效果理应相同

```shell
pip install paddlepaddle==2.4.1 pytest-runner paddlespeech==1.4.1 librosa==0.10.1 pymilvus pymysql transformers sentence_transformers LAC noisereduce pydub -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```bat
pip install paddlepaddle==2.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install paddlespeech==1.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install librosa==0.10.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pymilvus -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pymysql -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install sentence_transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install LAC -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pydub -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install noisereduce -i https://pypi.tuna.tsinghua.edu.cn/simple
```

[//]: # (```shell)

[//]: # (pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple)

[//]: # (```)
#### pytorch
- Windows:
```sh
pip install torch torchvision torchaudio
```
- Linux:
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 数据库配置（已部署服务器可忽略）

- 在`config.yaml`下配置内容

#### mysql

- 启动
- `.\data`目录在仓库中的`database\mysql`目录下，可以根据需要挂载到你自己的地址

```shell
cd database\mysql
```

```sh
docker run --privileged=true  -v .\data\:/var/lib/mysql -v .\logs\:/var/log/mysql -v .\conf\:/etc/mysql -v .\my.cnf:/etc/mysql/my.cnf  -p 8886:3306 --name mysql -e MYSQL_ROOT_PASSWORD=123456 -d mysql
```

- 可视化：Navicat等工具

#### milvus

- 启动
    - 首先切换到目录`Speech\database\milvus`下，可以看到一个docker-compose.yml文件，`docker compose up -d`即可

```shell
cd Speech\database\milvus
```

```
docker compose up -d
```

- 可视化：https://github.com/zilliztech/attu
