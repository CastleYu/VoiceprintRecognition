from sqlalchemy import QueuePool

from ._dao import MySQLDAO, Base
from .model import User, Command

DEBUG = False


class ConnectPoolConfig:
    poolclass = QueuePool,
    pool_size = 5
    max_overflow = 10
    pool_timeout = 30
    pool_recycle = 600


class MySQLClient:
    def __init__(self, host, port, user, password, database):
        self.user = MySQLDAO(User)
        self.command = MySQLDAO(Command)
        key_dict = {
            'host': host,
            'port': port,
            'user': user,
            'password': password,
            'database': database,
            'poolclass': ConnectPoolConfig.poolclass,
            "pool_size": ConnectPoolConfig.pool_size,
            "max_overflow": ConnectPoolConfig.max_overflow,
            "pool_timeout": ConnectPoolConfig.pool_timeout,
            "pool_recycle": ConnectPoolConfig.pool_recycle,
            "echo": DEBUG
        }
        self.user.connect(**key_dict)
        self.command.connect(**key_dict)

    def add_command(self, command, label="LAUNCH"):
        self.command.add(Command(action=command, label=label))

    def del_command(self, command):
        self.command.delete(self.command.get_one(action=command))

    def get_command_by_command(self, command):
        return self.command.get(action=command)

    def get_all_commands(self):
        return self.command.get_all()

    def get_command_by_label(self, label):
        return self.command.get(label=label)

    def add_user(self, username, voiceprint, permission_level):
        new_user = User(username=username, voiceprint=voiceprint, permission_level=permission_level)
        self.user.add(new_user)

    def del_user(self, id_):
        self.user.delete(self.user.get_one(id=id_))

    def get_user_by_id(self, id_):
        return self.user.get(id=id_)

    def get_all_users(self):
        return self.user.get_all()

    def update_user(self):
        self.user.commit()
