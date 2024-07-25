from flask import Flask, request, redirect, url_for, render_template, flash
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
import pymysql
import os
import yaml
import numpy as np
from werkzeug.utils import secure_filename
from asr import SpeechRecognitionAdapter, PaddleSpeechRecognition
from vector import SpeakerVerificationAdapter, PaddleSpeakerVerification
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # 用于闪现消息

DEFAULT_TABLE = "audio_table"
table_name = 'audio'

# 加载BERT模型和分词器
tokenizer = AutoTokenizer.from_pretrained('./bert-base-chinese')
model = AutoModel.from_pretrained('./bert-base-chinese')


def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

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
similarity_threshold = 0.8

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
    def create_action_table(self):
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS action (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            action VARCHAR(255) UNIQUE
        );
        """
        self.cursor.execute(create_table_sql)
        self.connection.commit()

    def insert_action(self, action):
        insert_sql = f"INSERT INTO action (action) VALUES (%s)"
        self.cursor.execute(insert_sql, (action,))
        self.connection.commit()

    def delete_action(self, action):
        delete_sql = f"DELETE FROM action WHERE action = %s"
        self.cursor.execute(delete_sql, (action,))
        self.connection.commit()

    def get_all_actions(self):
        select_sql = "SELECT action FROM action"
        self.cursor.execute(select_sql)
        results = self.cursor.fetchall()
        return [result[0] for result in results]


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

@app.route('/add_action', methods=['POST'])
def add_action():
    # 检查请求中是否包含指令部分
    if 'action' not in request.form:
        flash('No action part')
        return redirect(request.url)

    action = request.form['action']

    # 检查指令是否为空
    if action == '':
        flash('No action provided')
        return redirect(request.url)

    # 确保action表已经创建
    mysql_client.create_action_table()

    # 将指令插入到MySQL
    mysql_client.insert_action(action)

    return 'Action added successfully'

@app.route('/delete_action', methods=['POST'])
def delete_action():
    # 检查请求中是否包含指令部分
    if 'action' not in request.form:
        flash('No action part')
        return redirect(request.url)

    action = request.form['action']

    # 检查指令是否为空
    if action == '':
        flash('No action provided')
        return redirect(request.url)

    # 从MySQL中删除指令
    mysql_client.delete_action(action)

    return 'Action deleted successfully'

def match_action(action):
    # 获取所有指令
    all_actions = mysql_client.get_all_actions()

    # 计算相似度
    keyword_emb = get_embedding(action)
    similarities = [(action, cosine_similarity(keyword_emb, get_embedding(action))[0][0]) for action in all_actions]
    # similarities = [(action, np.dot(keyword_emb, get_embedding(action).T)[0][0]) for action in all_actions]

    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)

    if not similarities or similarities[0][1] < similarity_threshold:  # 添加阈值检查
        return 'No matching actions found'
    print(similarities[0][1])

    # 返回最匹配的指令
    best_match = similarities[0][0]
    return best_match

@app.route('/search_action', methods=['GET'])
def search_action():
    # 检查请求中是否包含指令部分
    if 'action' not in request.form:
        flash('No action part')
        return redirect(request.url)

    action = request.form['action']

    # 检查关键词是否为空
    if action == '':
        flash('No action provided')
        return redirect(request.url)

    best_match = match_action(action)
    return 'Best match action is: ' + best_match