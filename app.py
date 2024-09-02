import os.path
import sys
import traceback

import noisereduce as nr
import numpy as np
import yaml
from flask import Flask, request, redirect, flash, jsonify
from flask_cors import CORS
from pydub import AudioSegment
from scipy.io import wavfile

from action.action_matcher import *
from audio.asr import PaddleSpeechRecognition, SpeechRecognitionAdapter
from audio.vector import PaddleSpeakerVerification, SpeakerVerificationAdapter
from dao.milvus_dao import MilvusClient
from dao.mysql_dao import MySQLClient
from utils.file_utils import check_file_in_request, save_file

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # 用于闪现消息
CORS(app)


# 配置标准输出和标准错误的编码
def set_console_encoding(encoding='utf-8'):
    sys.stdout.reconfigure(encoding=encoding)
    sys.stderr.reconfigure(encoding=encoding)


DEFAULT_TABLE = "audio_table"
table_name = 'audio'
user_table = 'user'
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
milvus_client.create_collection(table_name)
mysql_client = MySQLClient(config['mysql']['host'], config['mysql']['port'], config['mysql']['user'],
                           config['mysql']['password'], config['mysql']['database'])

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'wav'}

paddleASR = SpeechRecognitionAdapter(PaddleSpeechRecognition())
paddleVector = SpeakerVerificationAdapter(PaddleSpeakerVerification())
models_path = os.path.abspath(os.path.join('action', 'bert_models'))
action_matcher = InstructionMatcher(models_path).load(MicomlMatcher('paraphrase-multilingual-MiniLM-L12-v2'))


# action_matcher = InstructionMatcher(models_path).load(GoogleMatcher('google-bert-base-chinese'))

def pre_process(audio_file):
    audio = AudioSegment.from_wav(audio_file)
    samples = np.array(audio.get_array_of_samples())

    # 获取音频文件的采样率
    rate, data = wavfile.read(audio_file)

    # 使用noisereduce库进行降噪
    reduced_noise = nr.reduce_noise(y=samples, sr=rate)

    # 将降噪后的数据转换回AudioSegment
    reduced_noise_audio = AudioSegment(
        reduced_noise.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )
    # # 降低采样率到16000 Hz
    # reduced_noise_audio = reduced_noise_audio.set_frame_rate(16000)

    output_dir = "./processed"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存降噪和降采样后的音频文件
    filename = os.path.basename(audio_file)
    output_file = os.path.join(output_dir, filename)
    reduced_noise_audio.export(output_file, format="wav")

    return output_file


@app.route('/load', methods=['PUT'])
def load():
    is_valid, message, files = check_file_in_request(request)
    if not is_valid:
        flash(message)
        return redirect(request.url)

    file_paths = []
    for file in files:
        file_path = save_file(file, app.config['UPLOAD_FOLDER'])
        file_paths.append(file_path)

    try:
        audio_embs = []
        for file_path in file_paths:
            pro_path = pre_process(file_path)
            audio_emb = paddleVector.get_embedding(pro_path)
            audio_embs.append(audio_emb)
        audio_embs_array = np.array(audio_embs)
        average_emb = np.mean(audio_embs_array, axis=0)

        # 将特征向量插入 Milvus 并获取 ID
        milvus_ids = milvus_client.insert(table_name, [average_emb.tolist()])
        milvus_client.create_index(table_name)

        # 将 ID 和音频信息存储到 MySQL
        username = request.form.get('username')
        permission_level = int(request.form.get('permission_level'))
        mysql_client.create_mysql_table(user_table)
        user_id = mysql_client.load_data_to_mysql(user_table, [(username, milvus_ids[0], permission_level)])

        response = {
            'result': SUCCESS,
            'data': {
                'user_id': user_id,
                'voiceprint': milvus_ids[0],
                'permission_level': permission_level
            }
        }
    except Exception as e:
        traceback.print_exc()
        response = {
            'result': FAILED,
            'data': {
                'error': type(e).__name__
            }
        }
    finally:
        for file_path in file_paths:
            os.remove(file_path)

    return jsonify(response)


@app.route('/delete_user', methods=['POST'])
def delete_user():
    # 检查请求中是否包含指令部分
    if 'user_id' not in request.form:
        flash('No user_id part')
        return redirect(request.url)

    user_id = request.form['user_id']

    # 检查指令是否为空
    if user_id == '':
        flash('No user_id provided')
        return redirect(request.url)

    # 从MySQL中删除指令
    voiceprint_id = mysql_client.get_voiceprint_by_id(user_id)
    milvus_client.delete_by_id(table_name, voiceprint_id)
    mysql_client.delete_user(user_id)

    return jsonify({'result': SUCCESS})


@app.route('/get_all_user', methods=['GET'])
def get_all_user():
    user_set = mysql_client.get_all_users()

    response = {
        'result': SUCCESS,
        'data': {
            "user_set": user_set
        }
    }
    return jsonify(response)


@app.route('/delete_all_user', methods=['GET'])
def delete_all_user():
    result = FAILED
    try:
        mysql_client.delete_all_users()
        milvus_client.delete_all(table_name)
        result = SUCCESS
    except Exception as e:
        traceback.print_exc()
    finally:
        return jsonify({"result": result})


@app.route('/update_user', methods=['POST'])
def update_user():
    username = request.form.get('username')
    permission_level = int(request.form.get('permission_level'))
    user_id = int(request.form.get('id'))

    mysql_client.update_user_info(user_table, user_id, username, permission_level)

    return jsonify({'result': SUCCESS})


@app.route('/asr', methods=['POST'])
def asr():
    is_valid, message, file = check_file_in_request(request)
    if not is_valid:
        flash(message)
        return redirect(request.url)

    file_path = save_file(file, app.config['UPLOAD_FOLDER'])
    try:
        text = paddleASR.recognize(file_path)
        if text:
            response = {
                'result': SUCCESS,
                'data': {
                    'text': text
                }
            }
        else:
            response = {
                'result': FAILED,
                'data': {
                    'error': 'Text not recognized'
                }
            }
    except Exception as e:
        traceback.print_exc()
        response = {
            'result': FAILED,
            'data': {
                'error': type(e).__name__
            }
        }
    finally:
        os.remove(file_path)

    return jsonify(response)


@app.route('/recognize', methods=['POST'])
def recognize():
    # 检查请求中的文件是否有效
    is_valid, message, files = check_file_in_request(request)
    if not is_valid:
        flash(message)
        return redirect(request.url)

    # 保存文件到指定路径
    file_path = save_file(files[0], app.config['UPLOAD_FOLDER'])

    try:
        pro_path = pre_process(file_path)
        # 获取音频嵌入向量
        audio_embedding = paddleVector.get_embedding(pro_path)

        # 在 Milvus 中搜索相似音频
        search_results = milvus_client.search(audio_embedding, table_name, 1)
        user_name = 'None'
        similar_distance = '0'
        similarity_score = '0'
        recognize_result = FAILED
        user_id = '0'
        asr_result = ''

        if search_results:
            user_id = str(search_results[0][0].id)
            user_name = mysql_client.find_user_name_by_id(user_id)
            similar_distance = search_results[0][0].distance
            similar_vector = np.array(search_results[0][0].entity.vec, dtype=np.float32)

            # 计算相似度评分
            similarity_score = paddleVector.get_embeddings_score(similar_vector, audio_embedding)

            # 根据相似度评分确定识别结果
            if similarity_score >= accuracy_threshold:
                recognize_result = SUCCESS
                asr_result = paddleASR.recognize(file_path)


        # 构建响应
        response = {
            'result': recognize_result,
            'data': {
                'username': user_name,
                'user_id': user_id,
                'similar_distance': similar_distance,
                'similarity_score': similarity_score,
                'asr_result': asr_result,
                'possible_action': do_search_action(asr_result)[1]
            }
        }
    except Exception as e:
        traceback.print_exc()
        response = {
            'result': FAILED,
            'data': {
                'error': type(e).__name__
            }
        }
    finally:
        # 删除临时文件
        os.remove(file_path)

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
    action_id, best_match, similarity_score = do_search_action(action)
    response = {
        'result': SUCCESS,
        'data': {
            "action_id": action_id,
            'best_match_action': best_match,
            'similarity_percent': f'{similarity_score * 100:.1f}'
        }
    }
    return jsonify(response)


def do_search_action(action):
    action_set = mysql_client.get_all_actions()
    best_match, similarity_score = action_matcher.match(action, action_set)
    action_id = mysql_client.get_action_id(best_match)
    return action_id, best_match, similarity_score


@app.route('/get_all_action', methods=['GET'])
def get_all_action():
    milvus_client.query_all(table_name)
    action_set = mysql_client.get_all_actions()

    response = {
        'result': SUCCESS,
        'data': {
            "action_set": action_set
        }
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run()
