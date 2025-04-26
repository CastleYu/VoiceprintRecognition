import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = 'uploads'
AUDIO_TABLE = 'audio'
USER_TABLE = 'user'


class Milvus:
    host = '47.121.186.134'
    port = '19530'


class MySQL:
    host = '47.121.186.134'
    port = 3306
    user = 'test'
    password = 'testpwd'
    database = 'voiceprint'


class Algorithm:
    threshold = 0.8
    low_text_threshold: 6      # 低文本量阈值（字符数）
    high_text_threshold: 12  # 高文本量阈值（字符数）

class Model:
    ModelDir = os.path.abspath(os.path.join(ROOT_DIR, 'action', 'bert_models'))
    ModelBinMap = {
        "google-bert-base-chinese": ["pytorch_model.bin"],
        "paraphrase-multilingual-MiniLM-L12-v2": ["0_Transformer", "pytorch_model.bin"]
    }
    # 模型名 和 路径
    DeepSpeakerModelDir = os.path.abspath(os.path.join(ROOT_DIR, 'model', 'ResCNN_triplet_training_checkpoint_265.h5'))


class Update:
    ModelDir = Model.ModelDir
    API = 'https://api.github.com/repos/CastleYu/VoiceprintRecognition/releases'
    RESOURCES_KEYWORD = 'resources'
