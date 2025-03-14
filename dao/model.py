from sqlalchemy import Column, Integer, String

from ._dao import Base


# ORM 模型定义
class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    username = Column(String(255))
    voiceprint = Column(String(64))
    permission_level = Column(Integer)


class Command(Base):
    __tablename__ = 'command'
    id = Column(Integer, primary_key=True)
    action = Column(String(255), unique=True)
    label = Column(String(255), default='LAUNCH', nullable=False)
    slot = Column(String(255))
