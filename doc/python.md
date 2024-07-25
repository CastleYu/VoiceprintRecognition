# Python环境安装

- python环境安装请遵照一下步骤，如果根据requirement.txt安装会出现包冲突

```
conda create --name speech python=3.8
pip install paddlepaddle==2.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install paddlespeech==1.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install librosa==0.10.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pymilvus -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pymysql -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple

# pytorch
Windows: pip install torch torchvision torchaudio
Linux: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

