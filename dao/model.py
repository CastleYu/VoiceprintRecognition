from pymilvus import FieldSchema, DataType
from sqlalchemy import Column, Integer, String

from ._dao import Base, Schema


# ORM 模型定义
class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    username = Column(String(255))
    voiceprint = Column(String(64))
    permission_level = Column(Integer)

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', voiceprint='{self.voiceprint}', permission_level={self.permission_level})>"


class Command(Base):
    __tablename__ = 'command'
    id = Column(Integer, primary_key=True)
    action = Column(String(255), unique=True)
    label = Column(String(255), default='LAUNCH', nullable=False)
    slot = Column(String(255))


class VoicePrint(Schema):
    __collection_name__ = "voiceprint"
    id = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    vec = FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=192)
