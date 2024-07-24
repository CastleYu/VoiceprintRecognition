from flask import Flask, request, redirect, url_for, render_template, flash
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
import pymysql
import os
import yaml
import numpy as np
from werkzeug.utils import secure_filename
from asr import SpeechRecognitionAdapter, PaddleSpeechRecognition
from vector import SpeakerVerificationAdapter, PaddleSpeakerVerification

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # 用于闪现消息

DEFAULT_TABLE = "audio_table"
table_name = 'audio'

# 读取配置文件
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

# 设置文件上传路径
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

accuracy_threshold = 0.8


# Milvus数据库
class MilvusClient:
    def __init__(self, host, port):
        connections.connect(alias='default', host=host, port=port)

    def create_collection(self, table_name):
        # 检查集合是否已经存在
        if utility.has_collection(table_name):
            return Collection(name=table_name)

        # 定义集合的 schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # 主键
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=192)
        ]
        schema = CollectionSchema(fields, description="audio collection")

        # 创建集合
        collection = Collection(name=table_name, schema=schema)
        return collection

    def insert(self, table_name, vectors):
        # 确保集合已经创建
        collection = self.create_collection(table_name)
        # 插入数据并返回 ID
        ids = collection.insert([vectors]).primary_keys  # 获取 ID
        return ids

    def create_index(self, table_name):
        collection = Collection(name=table_name)
        index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
        collection.create_index(field_name="vec", index_params=index_params)

    def search(self, query_vectors, top_k=1):
        # 确保集合已经创建
        collection = self.create_collection(table_name)
        # 搜索
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(data=[query_vectors], anns_field="vec", param=search_params, limit=top_k, output_fields=['vec'])
        return results


# Mysql数据库
class MySQLClient:
    def __init__(self, host, port, user, password, database):
        self.connection = pymysql.connect(host=host, port=port, user=user, password=password, database=None)
        self.cursor = self.connection.cursor()

        self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
        self.connection.select_db(database)

    def create_mysql_table(self, table_name):
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id BIGINT AUTO_INCREMENT  PRIMARY KEY,
            voiceprint BIGINT
        );
        """
        self.cursor.execute(create_table_sql)
        self.connection.commit()

    def load_data_to_mysql(self, table_name, data):
        insert_sql = f"INSERT INTO {table_name} (voiceprint) VALUES (%s)"
        self.cursor.executemany(insert_sql, data)
        self.connection.commit()


milvus_client = MilvusClient(config['milvus']['host'], config['milvus']['port'])
mysql_client = MySQLClient(config['mysql']['host'], config['mysql']['port'], config['mysql']['user'], config['mysql']['password'], config['mysql']['database'])


# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'wav'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_file_in_request():
    # 检查请求中是否包含文件部分
    if 'audio_file' not in request.files:
        flash('No file part')
        return False, None

    file = request.files['audio_file']

    # 检查文件是否被选择
    if file.filename == '':
        flash('No selected file')
        return False, None

    # 检查文件是否在允许的扩展名列表中
    if not (file and allowed_file(file.filename)):
        flash('File type not allowed')
        return False, None

    return True, file


paddleASR = SpeechRecognitionAdapter(PaddleSpeechRecognition())
paddleVector = SpeakerVerificationAdapter(PaddleSpeakerVerification())


@app.route('/load', methods=['PUT'])
def load():
    is_valid, file = check_file_in_request()
    if not is_valid:
        return redirect(request.url)

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    file.save(file_path)

    audio_emb = paddleVector.get_embedding(file_path)

    # 删除临时保存的文件
    os.remove(file_path)

    # 将特征向量插入 Milvus 并获取 ID
    milvus_ids = milvus_client.insert(table_name, [audio_emb.tolist()])
    milvus_client.create_index(table_name)

    # 将 ID 和音频信息存储到 MySQL
    mysql_client.create_mysql_table(table_name)
    mysql_client.load_data_to_mysql(table_name, milvus_ids)

    return str(milvus_ids)


@app.route('/asr', methods=['POST'])
def asr():
    is_valid, file = check_file_in_request()
    if not is_valid:
        return redirect(request.url)

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    file.save(file_path)

    text = paddleASR.recognize(file_path)

    # 删除临时保存的文件
    os.remove(file_path)

    return 'ASR Result: \n' + text

@app.route('/recognize', methods=['POST'])
def recognize():
    is_valid, file = check_file_in_request()
    if not is_valid:
        return redirect(request.url)

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    file.save(file_path)

    audio_emb = paddleVector.get_embedding(file_path)
    results = milvus_client.search(audio_emb, 1)
    user_id = results[0][0].id
    similar_distance = results[0][0].distance
    similar_vector = np.array(results[0][0].entity.vec, dtype=np.float32)

    similar = paddleVector.get_embeddings_score(similar_vector, audio_emb)
    if similar < accuracy_threshold:
        return "Illegal"

    text = paddleASR.recognize(file_path)

    # 删除临时保存的文件
    os.remove(file_path)

    return 'ASR Result: \n' + text
