import traceback
import tempfile
from typing import Optional

import librosa
import soundfile as sf
import numpy as np
import random
import logging

from flask import Flask, request, redirect, flash, jsonify
from flask_cors import CORS

import config

# Configure logging
logger = logging.getLogger(__name__)
SAMPLE_RATE = 16000  # Standard sample rate for voice processing
from action.action_matcher import *
from action.intent_recg import IntentRecognition
from audio.asr import PaddleSpeechRecognition, SpeechRecognitionAdapter
from audio.vector import PaddleSpeakerVerification, SpeakerVerificationAdapter, DeepSpeakerVerification
from config import AUDIO_TABLE, UPLOAD_FOLDER, ROOT_DIR
from const import SUCCESS, FAILED, HIGH_PRECISION_THRESHOLD
from dao import *
from utils.audioU import pre_process
from utils.fileU import check_file_in_request, save_file, create_abs_path, create_path
from utils.responseU import QuickResponse as qr

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # 用于闪现消息
CORS(app)
app.config['DEBUG'] = True

ACCURACY_THRESHOLD = config.Algorithm.threshold
LOW_TEXT_CHARACTER_LIMIT = config.Algorithm.low_text_threshold
HIGH_TEXT_CHARACTER_LIMIT = config.Algorithm.high_text_threshold
MODELS_DIR = config.Update.ModelDir

milvus_client = MilvusClient(config.Milvus.host, config.Milvus.port)
mysql_client = MySQLClient(config.MySQL.host, config.MySQL.port, config.MySQL.user,
                           config.MySQL.password, config.MySQL.database)
sqlite_client = SQLiteClient(create_path("data", "database.db"), False)
sql_client = mysql_client
paddleASR = SpeechRecognitionAdapter(PaddleSpeechRecognition())
paddleVector = SpeakerVerificationAdapter(PaddleSpeakerVerification())
deep_speakerVector = SpeakerVerificationAdapter(DeepSpeakerVerification())  # 初始化deep speaker模型

action_matcher = InstructionMatcher(MODELS_DIR).load(MicomlMatcher('paraphrase-multilingual-MiniLM-L12-v2'))
intent_recognizer = IntentRecognition(
    model_path='./action/bert_models/intent',
    intent_label_path='./action/bert_models/intent/data/SMP2019/intent_labels.txt',
    slot_label_path='./action/bert_models/intent/data/SMP2019/slot_labels.txt'
)


def do_search_action(action):
    # intent_result = intent_recognizer.detect_intent(action)
    # if isinstance(intent_result, dict):
    #     intent_data = intent_result
    # elif isinstance(intent_result, str):
    #     intent_data = json.loads(intent_result)
    # else:
    #     raise Exception(intent_result)
    # # label = intent_data.get('intent', 'LAUNCH')  # 获取意图识别后的标签

    command_objs = sql_client.get_all_commands()
    action_set = [(cmd.id, cmd.action) for cmd in command_objs]
    if action_set:
        # 提取 action 字段以便进行匹配
        actions = [action[1] for action in action_set]
        best_match, similarity_score = action_matcher.match(action, actions)

        # 处理相似度匹配结果
        if best_match != 'No matching actions found':
            # 找到最佳匹配的 action 对应的 ID
            action_id = next((act[0] for act in action_set if act[1] == best_match), None)
        else:
            action_id, similarity_score = None, 0
    else:
        action_id, best_match, similarity_score = None, 'No matching actions found', 0

    return action_id, best_match, similarity_score


@app.route('/load', methods=['PUT'])
def load():
    """
    处理用户上传的音频文件，进行声纹特征提取和注册

    流程:
    1. 验证上传文件有效性
    2. 分段处理音频文件并计算总时长
    3. 执行智能采样策略生成15秒的混合音频
    4. 提取声纹特征(192维和512维两种)
    5. 将特征向量存入Milvus向量数据库
    6. 创建用户记录并返回注册结果

    请求方法: PUT
    请求参数:
        - files: 音频文件(可多个)
        - username: 用户名
        - permission_level: 权限等级

    返回:
        - 成功: 包含用户ID、声纹ID等信息的JSON
        - 失败: 错误信息的JSON
    """
    # 1. 检查请求中的文件有效性
    is_valid, message, files = check_file_in_request(request)
    if not is_valid:
        return jsonify(qr.error(message))

    try:
        # 2. 音频文件预处理阶段
        segments = []  # 存储各音频片段信息
        total_duration = 0  # 总时长统计

        # 2.1 处理每个上传的音频文件
        for file in files:
            # 保存临时文件到上传目录
            file_path = save_file(file, UPLOAD_FOLDER)

            # 使用librosa加载音频并计算时长
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            total_duration += duration

            # 存储分段信息
            segments.append({
                'path': file_path,  # 文件路径
                'data': y,  # 音频数据
                'sr': sr,  # 采样率
                'duration': duration  # 时长(秒)
            })

        # 2.2 验证总时长是否满足最低要求(10秒)
        if total_duration < 10:
            raise ValueError(f"总时长不足10秒(当前{total_duration:.2f}秒)")

        # 3. 智能采样策略
        target_samples = 15 * SAMPLE_RATE  # 15秒的目标采样数
        sampled_audio = np.zeros(target_samples, dtype=np.float32)  # 初始化采样容器
        current_pos = 0  # 当前填充位置

        # 3.1 随机打乱片段顺序以增加样本多样性
        random.shuffle(segments)

        # 3.2 从各片段中智能选取音频片段
        for seg in segments:
            seg_samples = len(seg['data'])
            remaining_space = target_samples - current_pos

            if remaining_space <= 0:
                break

            # 计算本次要取的样本数(至少1秒，不超过剩余空间)
            take_samples = min(seg_samples, remaining_space,
                               max(int(seg['sr']), remaining_space // 2))

            # 随机选择起始位置
            start = random.randint(0, max(0, seg_samples - take_samples))

            # 填充采样数据
            sampled_audio[current_pos:current_pos + take_samples] = \
                seg['data'][start:start + take_samples]
            current_pos += take_samples

        # 4. 保存采样后的混合音频到临时文件
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, sampled_audio[:current_pos], SAMPLE_RATE)
        temp_file.close()

        # 5. 声纹特征提取
        # 5.1 预处理音频(降噪等)
        pro_path = pre_process(temp_file.name)
        # 5.2 提取两种维度的声纹特征
        emb_192 = paddleVector.get_embedding_from_file(pro_path)  # 192维特征
        emb_512 = deep_speakerVector.get_embedding_from_file(pro_path).squeeze()  # 512维特征

        # 6. 数据库操作
        # 6.1 将特征存入Milvus向量数据库
        milvus_ids = milvus_client.insert(AUDIO_TABLE, [emb_192.tolist()])
        milvus_client.insert_with_ids("deepSpeaker_vp512", [milvus_ids[0]], [emb_512.tolist()])

        # 6.2 创建用户记录
        username = request.form.get('username')
        permission_level = int(request.form.get('permission_level'))
        new_user = User(
            username=username,
            voiceprint=milvus_ids[0],  # 使用192维特征的ID
            permission_level=permission_level
        )
        sql_client.user.add(new_user)

        # 构造成功响应
        response = qr.data(
            user_id=new_user.id,
            voiceprint=milvus_ids[0],
            permission_level=permission_level,
            actual_duration=current_pos / SAMPLE_RATE  # 实际采样的时长
        )

    except Exception as e:
        # 异常处理
        response = qr.error(str(e))
        logger.error(f"注册失败: {str(e)}", exc_info=True)
    finally:
        # 7. 资源清理
        # 7.1 删除分段音频临时文件
        for seg in segments:
            if os.path.exists(seg['path']):
                os.remove(seg['path'])
        # 7.2 删除混合音频临时文件
        if 'temp_file' in locals() and os.path.exists(temp_file.name):
            os.remove(temp_file.name)

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
    user = sql_client.get_user_by_id(user_id)
    voiceprint_id = user.voiceprint if user else None
    if not voiceprint_id:
        print("no such user!")
    print(voiceprint_id)
    milvus_client.delete_by_id(AUDIO_TABLE, voiceprint_id)
    sql_client.del_user(user_id)

    return jsonify(qr.success())


@app.route('/get_all_user', methods=['GET'])
def get_all_user():
    users = sql_client.get_all_users()
    user_dict_list = [
        {
            'id': user.id,
            "username": user.username,
            'voiceprint': user.voiceprint,
            'permission_level': user.permission_level
        }
        for user in users
    ]
    response = qr.data(user_set=user_dict_list)
    return jsonify(response)


@app.route('/delete_all_user', methods=['GET'])
def delete_all_user():
    try:
        for user in sql_client.get_all_users():
            sql_client.del_user(user.id)
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

    user = sql_client.get_user_by_voiceprint(user_id)
    if user:
        user.username = username
        user.permission_level = permission_level
        sql_client.update_user()

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


@app.route('/recognize_legacy', methods=['POST'])
def recognize_legacy():
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
        audio_embedding = paddleVector.get_embedding_from_file(pro_path)

        # 在 Milvus 中搜索相似音频
        search_results = milvus_client.search(AUDIO_TABLE, audio_embedding, top_k=2)
        print(json.dumps(search_results, indent=4, ensure_ascii=False))
        user_name = 'None'
        similar_distance = '0'
        similarity_score = '0'
        recognize_result = FAILED
        voiceprint = '0'
        asr_result = ''

        if search_results:
            voiceprint = str(search_results[0][0].id)
            user = sql_client.get_user_by_voiceprint(voiceprint)
            user_name = user.username
            permission_level = user.permission_level
            similar_distance = search_results[0][0].distance
            similar_vector = np.array(search_results[0][0].entity.vec, dtype=np.float32)

            # 计算相似度评分
            similarity_score = paddleVector.get_embeddings_score(similar_vector, audio_embedding)

            # 根据相似度评分确定识别结果
            if (True or
                    similarity_score >= ACCURACY_THRESHOLD):
                recognize_result = SUCCESS
                asr_result = paddleASR.recognize(file_path)
                response = qr.result(
                    recognize_result,
                    username=user_name,
                    user_id=voiceprint,
                    permission_level=permission_level,
                    similar_distance=similar_distance,
                    similarity_score=similarity_score,
                    asr_result=asr_result,
                    possible_action=do_search_action(asr_result)[1]
                )
            else:
                response = qr.result(
                    recognize_result,
                    username=user_name,
                    user_id=voiceprint,
                    permission_level=permission_level,
                    similar_distance=similar_distance,
                    similarity_score=similarity_score,
                    error='声纹相似度不足'
                )
        else:
            response = qr.error('未找到近似声纹')
        # 构建响应

    except Exception as e:
        traceback.print_exc()
        response = qr.error(e)
    finally:
        # 删除临时文件
        os.remove(file_path)

    return jsonify(response)


@app.route('/recognizeAudioPrint', methods=['POST'])
def recognizeAudioPrint():
    # 检查请求中的文件是否有效
    is_valid, message, files = check_file_in_request(request)
    if not is_valid:
        flash(message)
        return redirect(request.url)

    # 保存文件到指定路径
    file_path = save_file(files[0], UPLOAD_FOLDER)
    try:
        # 计算音频时长和文本量
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        text_estimate = duration * (110 / 60)  # 取中间值110字/min

        pro_path = pre_process(file_path)

        # 根据文本量决定模型组合
        use_paddle = text_estimate >= LOW_TEXT_CHARACTER_LIMIT
        use_deep = text_estimate <= HIGH_TEXT_CHARACTER_LIMIT

        # 获取音频嵌入向量
        paddle_embedding = paddleVector.get_embedding_from_file(pro_path) if use_paddle else None
        deep_embedding = deep_speakerVector.get_embedding_from_file(pro_path).squeeze() if use_deep else None

        # 在 Milvus 中搜索相似音频
        paddle_results = milvus_client.search(AUDIO_TABLE, paddle_embedding, top_k=1) if use_paddle else []
        deep_results = milvus_client.search("deepSpeaker_vp512", deep_embedding, top_k=1) if use_deep else []

        user_name = 'None'
        recognize_result = FAILED
        VoicePrint_id = '0'
        user_id = '0'
        similarity_score = '0'

        # 判断识别结果
        if use_paddle and use_deep:
            # 使用两种模型的情况
            paddle_match = len(paddle_results) > 0 and paddleVector.get_embeddings_score(
                np.array(paddle_results[0][0].entity.vec, dtype=np.float32), paddle_embedding) >= ACCURACY_THRESHOLD
            deep_match = len(deep_results) > 0 and deep_speakerVector.get_embeddings_score(
                np.array(deep_results[0][0].entity.vec, dtype=np.float32)[np.newaxis, :],
                deep_embedding[np.newaxis, :]) >= ACCURACY_THRESHOLD

            if ACCURACY_THRESHOLD < HIGH_PRECISION_THRESHOLD:
                recognize_result = SUCCESS if paddle_match or deep_match else FAILED
            else:
                recognize_result = SUCCESS if paddle_match and deep_match else FAILED

            if recognize_result == SUCCESS:
                VoicePrint_id = str(paddle_results[0][0].id if paddle_match else deep_results[0][0].id)
                similarity_score = max(
                    paddleVector.get_embeddings_score(
                        np.array(paddle_results[0][0].entity.vec, dtype=np.float32),
                        paddle_embedding) if paddle_match else 0,
                    deep_speakerVector.get_embeddings_score(
                        np.array(deep_results[0][0].entity.vec, dtype=np.float32)[np.newaxis, :],
                        deep_embedding[np.newaxis, :]) if deep_match else 0
                )
        elif use_paddle:
            # 仅使用paddle模型
            if len(paddle_results) > 0:
                similarity_score = paddleVector.get_embeddings_score(
                    np.array(paddle_results[0][0].entity.vec, dtype=np.float32), paddle_embedding)
                recognize_result = SUCCESS if similarity_score >= ACCURACY_THRESHOLD else FAILED
                VoicePrint_id = str(paddle_results[0][0].id)
        elif use_deep:
            # 仅使用deep模型
            if len(deep_results) > 0:
                similarity_score = deep_speakerVector.get_embeddings_score(
                    np.array(deep_results[0][0].entity.vec, dtype=np.float32)[np.newaxis, :],
                    deep_embedding[np.newaxis, :])
                recognize_result = SUCCESS if similarity_score >= ACCURACY_THRESHOLD else FAILED
                VoicePrint_id = str(deep_results[0][0].id)

        # 获取用户信息
        if recognize_result == SUCCESS:
            user = sql_client.get_user_by_voiceprint(VoicePrint_id)
            user_name = user.username
            permission_level = user.permission_level

            response = qr.result(
                recognize_result,
                username=user_name,
                user_id=user_id,
                permission_level=permission_level,
                similarity_score=similarity_score,
                model_used='both' if use_paddle and use_deep else ('paddle' if use_paddle else 'deep')
            )
        else:
            response = qr.result(
                recognize_result,
                similarity_score=similarity_score,
                error='声纹识别失败',
                model_used='both' if use_paddle and use_deep else ('paddle' if use_paddle else 'deep')
            )

    except Exception as e:
        traceback.print_exc()
        response = qr.error(e)
    finally:
        # 删除临时文件
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
        # 计算音频时长和文本量
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        text_estimate = duration * (110 / 60)  # 取中间值110字/min

        pro_path = pre_process(file_path)

        # 根据文本量决定模型组合
        use_paddle = text_estimate >= LOW_TEXT_CHARACTER_LIMIT
        use_deep = text_estimate <= HIGH_TEXT_CHARACTER_LIMIT

        # 获取音频嵌入向量
        paddle_embedding = paddleVector.get_embedding_from_file(pro_path) if use_paddle else None
        deep_embedding = deep_speakerVector.get_embedding_from_file(pro_path).squeeze() if use_deep else None

        # 在 Milvus 中搜索相似音频
        paddle_results = milvus_client.search(AUDIO_TABLE, paddle_embedding, top_k=2) if use_paddle else []
        deep_results = milvus_client.search("deepSpeaker_vp512", deep_embedding, top_k=2) if use_deep else []

        print(paddle_results)
        print(deep_results)
        user_name = 'None'
        recognize_result = FAILED
        VoicePrint_id = '0'
        user_id = '0'
        similarity_score = '0'

        # 判断识别结果
        if use_paddle and use_deep:
            # 使用两种模型的情况
            paddle_match = len(paddle_results) > 0 and paddleVector.get_embeddings_score(
                np.array(paddle_results[0][0].entity.vec, dtype=np.float32), paddle_embedding) >= ACCURACY_THRESHOLD
            deep_match = len(deep_results) > 0 and deep_speakerVector.get_embeddings_score(
                np.array(deep_results[0][0].entity.vec, dtype=np.float32)[np.newaxis, :],
                deep_embedding[np.newaxis, :]) >= ACCURACY_THRESHOLD

            if ACCURACY_THRESHOLD < HIGH_PRECISION_THRESHOLD:
                recognize_result = SUCCESS if paddle_match or deep_match else FAILED
            else:
                recognize_result = SUCCESS if paddle_match and deep_match else FAILED

            if recognize_result == SUCCESS:
                VoicePrint_id = str(paddle_results[0][0].id if paddle_match else deep_results[0][0].id)
                similarity_score = max(
                    paddleVector.get_embeddings_score(
                        np.array(paddle_results[0][0].entity.vec, dtype=np.float32),
                        paddle_embedding) if paddle_match else 0,
                    deep_speakerVector.get_embeddings_score(
                        np.array(deep_results[0][0].entity.vec, dtype=np.float32)[np.newaxis, :],
                        deep_embedding[np.newaxis, :]) if deep_match else 0
                )
        elif use_paddle:
            # 仅使用paddle模型
            if len(paddle_results) > 0:
                similarity_score = paddleVector.get_embeddings_score(
                    np.array(paddle_results[0][0].entity.vec, dtype=np.float32), paddle_embedding)
                recognize_result = SUCCESS if similarity_score >= ACCURACY_THRESHOLD else FAILED
                VoicePrint_id = str(paddle_results[0][0].id)
        elif use_deep:
            # 仅使用deep模型
            if len(deep_results) > 0:
                similarity_score = deep_speakerVector.get_embeddings_score(
                    np.array(deep_results[0][0].entity.vec, dtype=np.float32)[np.newaxis, :],
                    deep_embedding[np.newaxis, :])
                similarity_score = similarity_score[0]
                recognize_result = SUCCESS if similarity_score >= ACCURACY_THRESHOLD else FAILED
                VoicePrint_id = str(deep_results[0][0].id)
        recognize_result = SUCCESS
        # 获取用户信息
        similarity_score = str(similarity_score)
        user = sql_client.get_user_by_voiceprint(VoicePrint_id)
        user_name = user.username
        permission_level = user.permission_level
        asr_result = paddleASR.recognize(file_path)
        possible_action = do_search_action(asr_result)[1]
        if recognize_result == SUCCESS:
            response = qr.result(
                recognize_result,
                username=user_name,
                user_id=user_id,
                permission_level=permission_level,
                similarity_score=similarity_score,
                model_used='both' if use_paddle and use_deep else ('paddle' if use_paddle else 'deep'),
                asr_result=asr_result,
                possible_action=possible_action
            )
        else:
            response = qr.result(
                recognize_result,
                username=user_name,
                user_id=user_id,
                permission_level=0,
                similarity_score=similarity_score,
                error='声纹识别失败',
                model_used='both' if use_paddle and use_deep else ('paddle' if use_paddle else 'deep')
            )

    except Exception as e:
        traceback.print_exc()
        response = qr.error(e)
    finally:
        # 删除临时文件
        os.remove(file_path)
    print(response)
    return jsonify(response)


@app.route('/wake', methods=['POST'])
def wake():
    file = request.files.get('file')
    if not file:
        return jsonify({
            'error': 'Missing file'}), 400
    file_path = save_file(file, UPLOAD_FOLDER)
    wake_text = request.form.get('wake_text')  # 获取传入的验证文本
    logging.debug(wake_text)
    try:
        pro_path = pre_process(file_path)
        wake_result = FAILED
        # 获取音频嵌入向量
        asr_result = paddleASR.recognize(file_path)
        if asr_result != wake_text:
            response = qr.result(
                wake_result,
                error='文本不匹配'
            )
        else:
            audio_embedding = paddleVector.get_embedding_from_file(pro_path)

            # 在 Milvus 中搜索相似音频
            search_results = milvus_client.search(AUDIO_TABLE, audio_embedding, top_k=1)
            similarity_score = '0'
            voiceprint = '0'

            if search_results:
                voiceprint = str(search_results[0][0].id)
                # 获取用户名字
                user = sql_client.get_user_by_voiceprint(voiceprint)
                similar_vector = np.array(search_results[0][0].entity.vec, dtype=np.float32)

                # 计算相似度评分
                similarity_score = paddleVector.get_embeddings_score(similar_vector, audio_embedding)

                # 根据相似度评分确定识别结果
                if similarity_score >= ACCURACY_THRESHOLD:
                    user = mysql_client.get_user_by_voiceprint(voiceprint)
                    wake_result = SUCCESS
                    response = qr.result(
                        wake_result,
                        recognized_text=asr_result,
                        user_name=user.username,
                    )
                else:
                    response = qr.result(
                        wake_result,
                        error='声纹精度不够'
                    )
            else:
                response = qr.error('未找到对应用户')
        # 构建响应

    except Exception as e:
        traceback.print_exc()
        response = qr.error(e)
    finally:
        # 删除临时文件
        os.remove(file_path)

    return jsonify(response)


@app.route('/add_action', methods=['POST'])
def add_action():
    """新增指令（含 action / level / label / slot）"""
    # ─── 1. 参数提取 ───
    action = request.form.get('action', '').strip()
    level = request.form.get('level', 1)
    label = request.form.get('label', 'LAUNCH').strip() or 'LAUNCH'
    slot = request.form.get('slot', '').strip()

    # ─── 2. 基本校验 ───
    if not action:
        return jsonify(qr.result(FAILED, error='action 不能为空')), 400
    try:
        level = int(level)
    except ValueError:
        return jsonify(qr.result(FAILED, error='level 必须为整数')), 400

    new_cmd = Command(action=action, level=level, label=label, slot=slot)
    sql_client.command.add(new_cmd)
    try:
        sql_client.command.commit()
    except:
        sql_client.command.session.rollback()
        return jsonify(qr.result(FAILED, error='添加异常')), 409

    return jsonify(qr.success()), 201


# ---------- /add_action 新实现结束 ----------


# ---------- /update_action 实现开始 ----------
@app.route('/update_action', methods=['POST'])
def update_action():
    """
    更新指令
    必填字段：id
    可选字段：action、level、label、slot
    · 若 action 重复，返回 409
    · 若找不到对应 id，返回 404
    · 成功返回更新后的完整记录
    """
    # ─── 1. 提取参数 ───
    cmd_id = request.form.get('id')
    if not cmd_id:
        return jsonify(qr.result(FAILED, error='id 不能为空')), 400
    try:
        cmd_id = int(cmd_id)
    except ValueError:
        return jsonify(qr.result(FAILED, error='id 必须为整数')), 400

    action = request.form.get('action')
    level = request.form.get('level')
    label = request.form.get('label')
    slot = request.form.get('slot')

    # ─── 2. 查询现有记录 ───
    cmd: Optional[Command] = sql_client.command.get_one(cmd_id)
    if cmd is None:
        return jsonify(qr.result(FAILED, error=f'未找到 id={cmd_id} 的指令')), 404

    # ─── 3. 字段更新（仅修改用户传入的字段）───
    if action is not None:
        action = action.strip()
        if not action:
            return jsonify(qr.result(FAILED, error='action 不能为空')), 400
        cmd.action = action

    if level is not None:
        try:
            cmd.level = int(level)
        except ValueError:
            return jsonify(qr.result(FAILED, error='level 必须为整数')), 400

    if label is not None:
        cmd.label = label.strip() or 'LAUNCH'

    if slot is not None:
        cmd.slot = slot.strip()

    # ─── 4. 提交事务 ───
    try:
        sql_client.command.commit()
    except:  # 主要用于处理 action 唯一索引冲突
        sql_client.command.session.rollback()
        return jsonify(qr.result(FAILED, error='action 已存在，更新失败')), 409

    # ─── 5. 返回成功结果 ───
    return jsonify(qr.success()), 200


# ---------- /update_action 实现结束 ----------


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
    sql_client.del_command(action)

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
    action_set = [{
        "id": cmd.id,
        "action": cmd.action,
        "level": cmd.level,
        "label": cmd.label,
        "slot": cmd.slot}
        for cmd in sql_client.get_all_commands()
    ]
    response = qr.data(action_set=action_set)
    return jsonify(response)


if __name__ == '__main__':
    app.run()

    # string = input("输入文本捏：")
    # while string:
    #     print(do_search_action(string))
    #     string = input("输入：")
