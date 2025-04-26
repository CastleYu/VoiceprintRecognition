import traceback

from flask import Flask, request, redirect, flash, jsonify
from flask_cors import CORS

import config
from action.action_matcher import *
from action.intent_recg import IntentRecognition
from audio.asr import PaddleSpeechRecognition, SpeechRecognitionAdapter
from audio.vector import PaddleSpeakerVerification, SpeakerVerificationAdapter, DeepSpeakerVerification
from config import AUDIO_TABLE, UPLOAD_FOLDER, ROOT_DIR
from const import SUCCESS, FAILED
from dao import *
from utils.audioU import pre_process
from utils.fileU import check_file_in_request, save_file, create_abs_path, create_path
from utils.responseU import QuickResponse as qr

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # 用于闪现消息
CORS(app)
app.config['DEBUG'] = True

ACCURACY_THRESHOLD = config.Algorithm.threshold
MODELS_DIR = config.Update.ModelDir

milvus_client = MilvusClient(config.Milvus.host, config.Milvus.port)
mysql_client = MySQLClient(config.MySQL.host, config.MySQL.port, config.MySQL.user,
                           config.MySQL.password, config.MySQL.database)
sqlite_client = SQLiteClient(create_path("data", "database.db"), False)
sql_client = mysql_client
paddleASR = SpeechRecognitionAdapter(PaddleSpeechRecognition())
paddleVector = SpeakerVerificationAdapter(PaddleSpeakerVerification())
deep_speakerVector = SpeakerVerificationAdapter(DeepSpeakerVerification)  # 初始化deep speaker模型

action_matcher = InstructionMatcher(MODELS_DIR).load(MicomlMatcher('paraphrase-multilingual-MiniLM-L12-v2'))
intent_recognizer = IntentRecognition(
    model_path='./action/bert_models/intent',
    intent_label_path='./action/bert_models/intent/data/SMP2019/intent_labels.txt',
    slot_label_path='./action/bert_models/intent/data/SMP2019/slot_labels.txt'
)


def do_search_action(action):
    intent_result = intent_recognizer.detect_intent(action)
    if isinstance(intent_result, dict):
        intent_data = intent_result
    elif isinstance(intent_result, str):
        intent_data = json.loads(intent_result)
    else:
        raise Exception(intent_result)
    label = intent_data.get('intent', 'LAUNCH')  # 获取意图识别后的标签

    command_objs = sql_client.get_command_by_label(label)
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
    is_valid, message, files = check_file_in_request(request)
    if not is_valid:
        flash(message)
        return redirect(request.url)

    file_paths = []
    for file in files:
        file_path = save_file(file, UPLOAD_FOLDER)
        file_paths.append(file_path)

    try:
        audio_embs_192 = []
        audio_embs_512 = []
        for file_path in file_paths:
            pro_path = pre_process(file_path)
            # 生成192维向量
            emb_192 = paddleVector.get_embedding_from_file(pro_path)
            audio_embs_192.append(emb_192)
            # 生成512维向量 (使用deep speaker模型)
            emb_512 = deep_speakerVector.get_embedding_from_file(pro_path)  # 假设模型有get_embedding方法
            audio_embs_512.append(emb_512)

        # 计算平均向量
        average_192 = np.mean(np.array(audio_embs_192), axis=0)
        average_512 = np.mean(np.array(audio_embs_512), axis=0)

        # 将192维特征向量插入 Milvus 并获取 ID
        milvus_ids = milvus_client.insert(AUDIO_TABLE, [average_192.tolist()])
        # 将512维特征向量插入到deepSpeaker_vp512集合，使用相同ID
        milvus_client.insert_with_ids("deepSpeaker_vp512", milvus_ids, [average_512])

        # 将 ID 和音频信息存储到 MySQL
        username = request.form.get('username')
        permission_level = int(request.form.get('permission_level'))
        new_user = User(username=username, voiceprint=milvus_ids[0], permission_level=permission_level)
        sql_client.user.add(new_user)
        user_id = new_user.id
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
    user = sql_client.get_user_by_id(user_id)
    voiceprint_id = user.voiceprint if user else None
    milvus_client.delete_by_id(AUDIO_TABLE, voiceprint_id)
    sql_client.del_user(user_id)

    return jsonify(qr.success())


@app.route('/get_all_user', methods=['GET'])
def get_all_user():
    user_set = sql_client.get_all_users()
    response = qr.data(user_set=user_set)
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

    user = sql_client.get_user_by_id(user_id)
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
    print(file_path)
    file_path = r'P:\xiangmu\python\Voice\test11.wav'
    try:
        text = paddleASR.recognize(file_path)
        print(text)
        text = '打开报表'
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
    print(file_path)
    file_path = r'P:\xiangmu\python\Voice\opendoor.wav'
    try:
        pro_path = pre_process(file_path)
        # 获取音频嵌入向量
        audio_embedding = paddleVector.get_embedding_from_file(pro_path)

        # 在 Milvus 中搜索相似音频
        search_results = milvus_client.search(AUDIO_TABLE, audio_embedding, top_k=1)
        user_name = 'None'
        similar_distance = '0'
        similarity_score = '0'
        recognize_result = FAILED
        user_id = '0'
        asr_result = ''

        if search_results:
            user_id = str(search_results[0][0].id)
            user = sql_client.get_user_by_id(user_id)
            user_name = user.username
            permission_level = user.permission_level
            similar_distance = search_results[0][0].distance
            similar_vector = np.array(search_results[0][0].entity.vec, dtype=np.float32)

            # 计算相似度评分
            similarity_score = paddleVector.get_embeddings_score(similar_vector, audio_embedding)

            # 根据相似度评分确定识别结果
            if similarity_score >= ACCURACY_THRESHOLD:
                recognize_result = SUCCESS
                asr_result = paddleASR.recognize(file_path)
                response = qr.result(
                    recognize_result,
                    username=user_name,
                    user_id=user_id,
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
                    user_id=user_id,
                    permission_level=permission_level,
                    similar_distance=similar_distance,
                    similarity_score=similarity_score,
                    error='相似度不够'
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
    print(file_path)
    file_path = r'P:\xiangmu\python\Voice\opendoor.wav'
    try:
        pro_path = pre_process(file_path)
        # 获取音频嵌入向量
        audio_embedding = paddleVector.get_embedding_from_file(pro_path)

        # 在 Milvus 中搜索相似音频
        search_results = milvus_client.search(AUDIO_TABLE, audio_embedding, top_k=1)
        user_name = 'None'
        similar_distance = '0'
        similarity_score = '0'
        recognize_result = FAILED
        user_id = '0'

        if search_results:
            user_id = str(search_results[0][0].id)
            user = sql_client.get_user_by_id(user_id)
            user_name = user.username
            permission_level = user.permission_level
            similar_distance = search_results[0][0].distance
            similar_vector = np.array(search_results[0][0].entity.vec, dtype=np.float32)

            # 计算相似度评分
            similarity_score = paddleVector.get_embeddings_score(similar_vector, audio_embedding)

            # 根据相似度评分确定识别结果
            if similarity_score >= ACCURACY_THRESHOLD:
                recognize_result = SUCCESS
                response = qr.result(
                    recognize_result,
                    username=user_name,
                    user_id=user_id,
                    permission_level=permission_level,
                    similar_distance=similar_distance,
                    similarity_score=similarity_score
                )
            else:
                response = qr.result(
                    recognize_result,
                    username=user_name,
                    user_id=user_id,
                    permission_level=permission_level,
                    similar_distance=similar_distance,
                    similarity_score=similarity_score,
                    error='相似度不够'
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


@app.route('/wake', methods=['POST'])
def wake():
    file = request.files.get('file')
    print(file)
    if not file:
        return jsonify({
            'error': 'Missing file'}), 400
    file_path = save_file(file, UPLOAD_FOLDER)
    wake_text = request.form.get('wake_text')  # 获取传入的验证文本

    print(f"接收到的唤醒文本: {wake_text}")
    print(f"接收到的文件路径: {file_path}")
    wake_text = "打开报表"
    file_path = r'P:\xiangmu\python\Voice\test11.wav'
    print(file_path)
    try:
        pro_path = pre_process(file_path)
        wake_result = FAILED
        # 获取音频嵌入向量
        asr_result = paddleASR.recognize(file_path)
        print(asr_result)
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
            user_id = '0'

            if search_results:
                user_id = str(search_results[0][0].id)
                #获取用户名字
                user = sql_client.get_user_by_id(user_id)
                print(user)
                similar_vector = np.array(search_results[0][0].entity.vec, dtype=np.float32)

                # 计算相似度评分
                similarity_score = paddleVector.get_embeddings_score(similar_vector, audio_embedding)

                # 根据相似度评分确定识别结果
                print(f'{similarity_score} > {ACCURACY_THRESHOLD} = {similarity_score >= ACCURACY_THRESHOLD}')
                if similarity_score >= ACCURACY_THRESHOLD:
                    wake_result = SUCCESS
                    response = qr.result(
                        wake_result,
                        recognized_text=asr_result,
                        user_id=user_id,
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
        # os.remove(file_path)
        pass

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

    # 将指令插入到MySQL
    sql_client.add_command(action)

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
    action_set = [cmd.action for cmd in sql_client.get_all_commands()]

    response = qr.data(action_set=action_set)
    return jsonify(response)


if __name__ == '__main__':
    app.run()