import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = 'uploads'
AUDIO_TABLE = 'audio'
USER_TABLE = 'user'


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
    threshold = 0.75111


class Model:
    ModelDir = os.path.abspath(os.path.join(ROOT_DIR, 'action', 'bert_models'))
    ModelBinMap = {
        "google-bert-base-chinese": ["pytorch_model.bin"],
        "paraphrase-multilingual-MiniLM-L12-v2": ["0_Transformer", "pytorch_model.bin"]
    }


class Update:
    ModelDir = Model.ModelDir
    API = 'https://api.github.com/repos/CastleYu/VoiceprintRecognition/releases'
    RESOURCES_KEYWORD = 'resources'
