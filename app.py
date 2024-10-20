import os.path
import traceback

import numpy as np
from flask import Flask, request, redirect, flash, jsonify
from flask_cors import CORS

import config
from action.action_matcher import *
from audio.asr import PaddleSpeechRecognition, SpeechRecognitionAdapter
from audio.vector import PaddleSpeakerVerification, SpeakerVerificationAdapter
from config import AUDIO_TABLE, USER_TABLE, UPLOAD_FOLDER
from const import SUCCESS, FAILED
from dao.milvus_dao import MilvusClient
from dao.mysql_dao import MySQLClient
from utils.audioU import pre_process
from utils.fileU import check_file_in_request, save_file
from utils.responseU import QuickResponse as qr

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # 用于闪现消息
CORS(app)

ACCURACY_THRESHOLD = config.Algorithm.threshold
MODELS_DIR = config.Update.ModelDir

milvus_client = MilvusClient(config.Milvus.host, config.Milvus.port)
milvus_client.create_collection(AUDIO_TABLE)
mysql_client = MySQLClient(config.MySQL.host, config.MySQL.port, config.MySQL.user,
                           config.MySQL.password, config.MySQL.database)

paddleASR = SpeechRecognitionAdapter(PaddleSpeechRecognition())
paddleVector = SpeakerVerificationAdapter(PaddleSpeakerVerification())

action_matcher = InstructionMatcher(MODELS_DIR).load(MicomlMatcher('paraphrase-multilingual-MiniLM-L12-v2'))


def do_search_action(action):
    action_set = mysql_client.get_all_actions()
    best_match, similarity_score = action_matcher.match(action, action_set)
    action_id = mysql_client.get_action_id(best_match)
    return action_id, best_match, similarity_score


@app.route('/load', methods=['PUT'])
def load():
    is_valid, message, files = check_file_in_request(request)
    if not is_valid:
        flash(message)
        return redirect(request.url)

    file_paths = []
    for file in files:
        file_path = save_file(file, UPLOAD_FOLDER)
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
        milvus_ids = milvus_client.insert(AUDIO_TABLE, [average_emb.tolist()])
        milvus_client.create_index(AUDIO_TABLE)

        # 将 ID 和音频信息存储到 MySQL
        username = request.form.get('username')
        permission_level = int(request.form.get('permission_level'))
        mysql_client.create_mysql_table(USER_TABLE)
        user_id = mysql_client.load_data_to_mysql(USER_TABLE, [(username, milvus_ids[0], permission_level)])

        response = qr.data(
            user_id=user_id,
            voiceprint=milvus_ids[0],
            permission_level=permission_level
        )
    except Exception as e:
        traceback.print_exc()
        response = qr.error(e)
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
    milvus_client.delete_by_id(AUDIO_TABLE, voiceprint_id)
    mysql_client.delete_user(user_id)

    return jsonify(qr.success())


@app.route('/get_all_user', methods=['GET'])
def get_all_user():
    user_set = mysql_client.get_all_users()
    response = qr.data(user_set=user_set)
    return jsonify(response)


@app.route('/delete_all_user', methods=['GET'])
def delete_all_user():
    try:
        mysql_client.delete_all_users()
        milvus_client.delete_all(AUDIO_TABLE)
        return jsonify(qr.success())
    except Exception as e:
        traceback.print_exc()
        return jsonify(qr.error(e))


@app.route('/update_user', methods=['POST'])
def update_user():
    username = request.form.get('username')
    permission_level = int(request.form.get('permission_level'))
    user_id = int(request.form.get('id'))

    mysql_client.update_user_info(USER_TABLE, user_id, username, permission_level)

    return jsonify(qr.success())


@app.route('/asr', methods=['POST'])
def asr():
    is_valid, message, file = check_file_in_request(request)
    if not is_valid:
        flash(message)
        return redirect(request.url)

    file_path = save_file(file, UPLOAD_FOLDER)
    try:
        text = paddleASR.recognize(file_path)
        if text:
            response = qr.data(text=text)
        else:
            response = qr.error("Text not recognized")
    except Exception as e:
        traceback.print_exc()
        response = qr.error(e)
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
    file_path = save_file(files[0], UPLOAD_FOLDER)

    try:
        pro_path = pre_process(file_path)
        # 获取音频嵌入向量
        audio_embedding = paddleVector.get_embedding(pro_path)

        # 在 Milvus 中搜索相似音频
        search_results = milvus_client.search(audio_embedding, AUDIO_TABLE, 1)
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
            if similarity_score >= ACCURACY_THRESHOLD:
                recognize_result = SUCCESS
                asr_result = paddleASR.recognize(file_path)

        # 构建响应

        response = qr.result(
            recognize_result,
            username=user_name,
            user_id=user_id,
            similar_distance=similar_distance,
            similarity_score=similarity_score,
            asr_result=asr_result,
            possible_action=do_search_action(asr_result)[1]
        )
    except Exception as e:
        traceback.print_exc()
        response = qr.error(e)
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

    return jsonify(qr.success())


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

    return jsonify(qr.success())


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
    response = qr.data(
        action_id=action_id,
        best_match_action=best_match,
        similarity_percent=f'{similarity_score * 100:.1f}'
    )
    return jsonify(response)


@app.route('/get_all_action', methods=['GET'])
def get_all_action():
    milvus_client.query_all(AUDIO_TABLE)
    action_set = mysql_client.get_all_actions()

    response = qr.data(action_set=action_set)
    return jsonify(response)


if __name__ == '__main__':
    app.run()
