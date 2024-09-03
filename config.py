import os


class Milvus:
    host = '120.24.44.238'
    port = '19530'


class MySQL:
    host = '120.24.44.238'
    port = 3306
    user = 'test'
    password = 'testpwd'
    database = 'voiceprint'


class Algorithm:
    threshold = 0.8


ModelDir = os.path.abspath(os.path.join('action', 'bert_models'))
