import csv
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# -------------------------
# 定义数据库连接配置
# -------------------------
from config import MySQL

# -------------------------
# 创建数据库引擎和会话
# -------------------------
db_url = (
    f"mysql+pymysql://{MySQL.user}:{MySQL.password}"
    f"@{MySQL.host}:{MySQL.port}/{MySQL.database}"
)
engine = create_engine(db_url, echo=True)
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

# -------------------------
# 定义数据库表对应的模型
# -------------------------
Base = declarative_base()


class Action(Base):
    __tablename__ = 'action'
    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String(255), unique=True)
    label = Column(String(255), default='LAUNCH', nullable=False)
    slot = Column(String(255))  # 虽然模型里有这个字段，但我们读取时会忽略


# 如果数据表还没创建，可以执行这行来建表
Base.metadata.create_all(engine)


# -------------------------
# 从 CSV 读取并写入数据库
# -------------------------
def insert_csv_to_db(csv_file_path: str):
    """
    从给定的 CSV 中读取 action、label 字段，写入 action 表，忽略 slot 字段。
    """
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 假设 CSV 确实含有 action、label、slot 等列
            action_val = row.get('action')
            label_val = row.get('label')
            # 忽略 slot，故不在此使用 row.get('slot')

            # 如果 CSV 里缺少某些字段，可以做相应的判断或默认处理
            if not action_val:
                # 如果没有 action, 可以选择跳过该条数据或做其他处理
                continue
            if not label_val:
                # 如果 label 为空，选择补默认值
                label_val = 'LAUNCH'

            new_action = Action(
                action=action_val,
                label=label_val
            )
            session.add(new_action)
    session.commit()


if __name__ == "__main__":
    # 调用插入函数，将 data.csv 数据导入数据库
    csv_path = "data.csv"  # 这里替换成实际 CSV 文件的路径
    insert_csv_to_db(csv_path)
