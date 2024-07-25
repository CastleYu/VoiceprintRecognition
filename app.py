import os.path

import numpy as np
import yaml
from flask import Flask, request, redirect, flash
from werkzeug.utils import secure_filename

from action.action_matcher import *
from audio.asr import PaddleSpeechRecognition, SpeechRecognitionAdapter
from audio.vector import PaddleSpeakerVerification, SpeakerVerificationAdapter
from dao.milvus_dao import MilvusClient
from dao.mysql_dao import MySQLClient

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
similarity_threshold = 0.8

milvus_client = MilvusClient(config['milvus']['host'], config['milvus']['port'])
mysql_client = MySQLClient(config['mysql']['host'], config['mysql']['port'], config['mysql']['user'],
                           config['mysql']['password'], config['mysql']['database'])

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'wav'}

paddleASR = SpeechRecognitionAdapter(PaddleSpeechRecognition())
paddleVector = SpeakerVerificationAdapter(PaddleSpeakerVerification())
action_set = mysql_client.get_all_actions()
models_path = os.path.abspath(os.path.join('action', 'bert_models'))
action_matcher = InstructionMatcher(models_path).load(MicomlMatcher('paraphrase-multilingual-MiniLM-L12-v2'))
# action_matcher = InstructionMatcher(models_path).load(GoogleMatcher('google-bert-base-chinese'))


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
    results = milvus_client.search(audio_emb, table_name, 1)
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

    best_match = action_matcher.match(action, action_set)
    return 'Best match action is: ' + best_match


if __name__ == '__main__':
    app.run()
