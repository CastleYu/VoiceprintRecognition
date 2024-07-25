import os.path
import sys

import numpy as np
import yaml
from flask import Flask, request, redirect, flash, jsonify

from action.action_matcher import *
from audio.asr import PaddleSpeechRecognition, SpeechRecognitionAdapter
from audio.vector import PaddleSpeakerVerification, SpeakerVerificationAdapter
from dao.milvus_dao import MilvusClient
from dao.mysql_dao import MySQLClient
from utils.file_utils import check_file_in_request, save_file

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # 用于闪现消息


# 配置标准输出和标准错误的编码
def set_console_encoding(encoding='utf-8'):
    sys.stdout.reconfigure(encoding=encoding)
    sys.stderr.reconfigure(encoding=encoding)


DEFAULT_TABLE = "audio_table"
table_name = 'audio'
SUCCESS = 'Success'
FAILED = 'Failed'


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


@app.route('/load', methods=['PUT'])
def load():
    is_valid, message, file = check_file_in_request(request)
    if not is_valid:
        flash(message)
        return redirect(request.url)

    file_path = save_file(file, app.config['UPLOAD_FOLDER'])

    audio_emb = paddleVector.get_embedding(file_path)

    os.remove(file_path)

    # 将特征向量插入 Milvus 并获取 ID
    milvus_ids = milvus_client.insert(table_name, [audio_emb.tolist()])
    milvus_client.create_index(table_name)

    # 将 ID 和音频信息存储到 MySQL
    mysql_client.create_mysql_table(table_name)
    mysql_client.load_data_to_mysql(table_name, milvus_ids)

    response = {
        'result': SUCCESS,
        'data': {
            'user_id': 0,
            'milvus_ids': milvus_ids,
            'voiceprint': milvus_ids
        }
    }
    return jsonify(response)


@app.route('/asr', methods=['POST'])
def asr():
    is_valid, message, file = check_file_in_request(request)
    if not is_valid:
        flash(message)
        return redirect(request.url)

    file_path = save_file(file, app.config['UPLOAD_FOLDER'])
    try:
        text = paddleASR.recognize(file_path)
    except Exception as e:
        text = None

    os.remove(file_path)

    response = {
        'result': SUCCESS if text else FAILED,
        'data': {
            ''
        }
    }
    return jsonify(response)


@app.route('/recognize', methods=['POST'])
def recognize():
    is_valid, message, file = check_file_in_request(request)
    if not is_valid:
        flash(message)
        return redirect(request.url)

    file_path = save_file(file, app.config['UPLOAD_FOLDER'])

    audio_emb = paddleVector.get_embedding(file_path)
    results = milvus_client.search(audio_emb, table_name, 1)
    user_id = results[0][0].id
    similar_distance = results[0][0].distance
    similar_vector = np.array(results[0][0].entity.vec, dtype=np.float32)

    similar = paddleVector.get_embeddings_score(similar_vector, audio_emb)
    recognize_result = FAILED
    if similar >= accuracy_threshold:
        recognize_result = SUCCESS
        text = paddleASR.recognize(file_path)
    else:
        text = ''

    os.remove(file_path)

    response = {
        'result': recognize_result,
        'data': {
            'user_id': user_id,
            'similar_distance': similar_distance,
            'similarity_score': similar,
            'asr_result': text
        }
    }

    return jsonify(response)


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

    return jsonify({'result': SUCCESS})


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

    return jsonify({'result': SUCCESS})


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

    best_match, similarity_score = action_matcher.match(action, action_set)
    action_id = mysql_client.get_action_id(best_match)
    response = {
        'result': SUCCESS,
        'data': {
            "action_id": action_id,
            'best_match_action': best_match,
            'similarity_percent': f'{similarity_score * 100:.1f}'
        }
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run()
