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


class Model:
    ModelDir = os.path.abspath(os.path.join(ROOT_DIR, 'action', 'bert_models'))
    ModelBinMap = {
        "google-bert-base-chinese": ["pytorch_model.bin"],
        "paraphrase-multilingual-MiniLM-L12-v2": ["0_Transformer", "pytorch_model.bin"]
    }
    # 模型名 和 路径


class Update:
    ModelDir = Model.ModelDir
    API = 'https://api.github.com/repos/CastleYu/VoiceprintRecognition/releases'
    RESOURCES_KEYWORD = 'resources'


if __name__ == '__main__':
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import OperationalError, InterfaceError

    connection_url = f"mysql+pymysql://{MySQL.user}:{MySQL.password}@{MySQL.host}:{MySQL.port}/{MySQL.database}"

    try:
        # 创建数据库引擎
        engine = create_engine(
            connection_url,
            connect_args={
                'connect_timeout': 5}  # 设置连接超时时间
        )

        # 测试连接
        with engine.connect() as connection:
            print("✅ 数据库连接成功！")
            result = connection.execute(text("SELECT VERSION()"))
            print(f"MySQL 服务器版本: {result.scalar()}")

    except ModuleNotFoundError:
        print("❌ 缺少依赖包，请安装 pymysql: pip install pymysql")
    except OperationalError as e:
        print(f"❌ 连接失败，错误信息: {e}")
        print("请检查：")
        print(f"1. 主机地址是否正确（当前值: {MySQL.host}）")
        print(f"2. 是否已启动MySQL服务（端口: {MySQL.port}）")
        print("3. 用户名/密码是否正确")
        print("4. 网络连接是否正常")
    except InterfaceError as e:
        print(f"❌ 接口错误: {e}")
    except Exception as e:
        print(f"❌ 未知错误: {e}")

if __name__ == "__main__":
    from pymilvus import connections, utility
    from pymilvus.exceptions import MilvusException

    try:
        # 尝试建立数据库连接（设置超时参数）
        connections.connect(
            host=Milvus.host,
            port=Milvus.port,
        )

        # 验证连接有效性（获取服务器版本）
        version = utility.get_server_version()
        print(f"✅ 成功连接到 Milvus 服务器")
        print(f"Milvus 服务器版本: {version}")
    except ModuleNotFoundError:
        print("❌ 缺少依赖库，请安装 pymilvus: pip install pymilvus")
    except MilvusException as e:
        error_code = e.code
        error_msg = e.message

        # 根据错误类型给出针对性提示
        if "connection attempt failed" in error_msg.lower():
            print(f"❌ 连接失败（错误码 {error_code}）")
            print("请检查：")
            print(f"1. 主机地址是否正确（当前值: {Milvus.host}）")
            print(f"2. 端口是否开放（当前值: {Milvus.port}）")
            print("3. 是否已启动Milvus服务（docker容器是否运行）")
            print("4. 防火墙设置（是否阻止了网络连接）")
        elif "timeout" in error_msg.lower():
            print(f"❌ 连接超时")
            print("可能原因：")
            print("1. 网络延迟过高或带宽不足")
            print("2. Milvus服务未正常启动")
            print("3. 服务器资源过载（CPU/内存不足）")
        else:
            print(f"❌ 连接错误（{error_code}）: {error_msg}")
    except Exception as e:
        print(f"❌ 未知错误: {str(e)}")
    finally:
        try:
            connections.disconnect("default")
        except:
            pass
