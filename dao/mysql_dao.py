from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, select, delete, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker


class MySQLClient:
    def __init__(self, host, port, user, password, database):
        self.database_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        self.engine = create_engine(self.database_url)
        self.Session = sessionmaker(bind=self.engine)
        self.metadata = MetaData()

        # Define table metadata for reflection
        self.user_table = Table('user', self.metadata,
                                Column('id', Integer, primary_key=True),
                                Column('username', String(255)),
                                Column('voiceprint', String(64)),
                                Column('permission_level', Integer))

        self.action_table = Table('action', self.metadata,
                                  Column('id', Integer, primary_key=True),
                                  Column('action', String(255), unique=True),
                                  Column('label', String(255), default='LAUNCH', nullable=False),
                                  Column('slot', String(255)),
                                  )

    def create_mysql_table(self, table_name):
        if table_name == 'user':
            self.user_table.create(self.engine, checkfirst=True)
        elif table_name == 'action':
            self.action_table.create(self.engine, checkfirst=True)

    def load_data_to_mysql(self, table_name, data_list):
        """

        :param table_name:
        :param data_list: 一个 （用户名，声纹，权限） 的三元组的列表
        :return:
        """
        connection = self.engine.connect()
        if table_name == 'user':
            table = self.user_table
        else:
            return
        try:
            formatted_data = [
                {
                    'username': entry[0],
                    'voiceprint': entry[1],
                    'permission_level': entry[2]
                }
                for entry in data_list
            ]
            connection.execute(table.insert(), formatted_data)
            connection.commit()
        except SQLAlchemyError as e:
            connection.rollback()
            print(f"MYSQL Error occurred: {e}\n occurred in load_data_to_mysql")
        finally:
            connection.close()

    def insert_action(self, action):
        connection = self.engine.connect()
        try:
            connection.execute(self.action_table.insert(), {
                'action': action})
            connection.commit()
        except SQLAlchemyError as e:
            connection.rollback()
            print(f"MYSQL Error occurred: {e}\n occurred in insert_action")
        finally:
            connection.close()

    def delete_action(self, action):
        connection = self.engine.connect()
        try:
            connection.execute(delete(self.action_table).where(self.action_table.c.action == action))
            connection.commit()
        except SQLAlchemyError as e:
            connection.rollback()
            print(f"MYSQL Error occurred: {e}\n occurred in delete_action")
        finally:
            connection.close()

    def get_all_actions(self):
        connection = self.engine.connect()
        try:
            result = connection.execute(select(self.action_table.c.action)).fetchall()
            return [row[0] for row in result]
        except SQLAlchemyError as e:
            print(f"MYSQL Error occurred: {e} \n occurred in get_all_actions")
            return []
        finally:
            connection.close()

    def get_action_id(self, action):
        connection = self.engine.connect()
        try:
            result = connection.execute(
                select(self.action_table.c.id).where(self.action_table.c.action == action)).fetchone()
            return result[0] if result else None
        except SQLAlchemyError as e:
            print(f"MYSQL Error occurred: {e} \n occurred in get_action_id")
            return None
        finally:
            connection.close()

    def delete_user(self, user_id):
        connection = self.engine.connect()
        try:
            connection.execute(delete(self.user_table).where(self.user_table.c.id == user_id))
            connection.commit()
        except SQLAlchemyError as e:
            connection.rollback()
            print(f"MYSQL Error occurred: {e} \n occurred in delete_user")
        finally:
            connection.close()

    def delete_all_users(self):
        connection = self.engine.connect()
        try:
            connection.execute(delete(self.user_table))
            connection.commit()
        except SQLAlchemyError as e:
            connection.rollback()
            print(f"MYSQL Error occurred: {e} \n occurred in delete_all_users")
        finally:
            connection.close()

    def get_all_users(self):
        connection = self.engine.connect()
        try:
            result = connection.execute(select(self.user_table)).fetchall()
            users = []
            for row in result:
                users.append({
                    "id": row[0],
                    "username": row[1],
                    "voiceprint": row[2],
                    "permission_level": row[3]
                })
            return users
        except SQLAlchemyError as e:
            print(f"MYSQL Error occurred: {e} \n occurred in get_all_users")
            return []
        finally:
            connection.close()

    def find_user_name_by_id(self, query_id):
        connection = self.engine.connect()
        try:
            result = connection.execute(
                select(self.user_table.c.username).where(self.user_table.c.voiceprint == query_id)).fetchone()
            return result[0] if result else None
        except SQLAlchemyError as e:
            print(f"MYSQL Error occurred: {e} \n occurred in find_permission_level_by_id")
            return None
        finally:
            connection.close()

    def find_permission_level_by_id(self, query_id):
        connection = self.engine.connect()
        try:
            result = connection.execute(
                select(self.user_table.c.permission_level).where(self.user_table.c.voiceprint == query_id)).fetchone()
            return result[0] if result else None
        except SQLAlchemyError as e:
            print(f"MYSQL Error occurred: {e} \n occurred in find_user_name_by_id")
            return None
        finally:
            connection.close()

    def update_user_info(self, table_name, user_id, username=None, permission_level=None):
        if table_name == 'user':
            table = self.user_table
        else:
            return

        update_fields = {}
        if username is not None:
            update_fields['username'] = username
        if permission_level is not None:
            update_fields['permission_level'] = permission_level

        if not update_fields:
            return

        connection = self.engine.connect()
        try:
            connection.execute(update(table).where(table.c.id == user_id).values(**update_fields))
            connection.commit()
        except SQLAlchemyError as e:
            connection.rollback()
            print(f"MYSQL Error occurred: {e} \n occurred in update_user_info")
        finally:
            connection.close()

    def get_voiceprint_by_id(self, user_id):
        connection = self.engine.connect()
        try:
            result = connection.execute(
                select(self.user_table.c.voiceprint).where(self.user_table.c.id == user_id)).fetchone()
            return result[0] if result else None
        except SQLAlchemyError as e:
            print(f"MYSQL Error occurred: {e} \n occurred in get_voiceprint_by_id")
            return None
        finally:
            connection.close()

    def get_actions_by_label(self, label):
        connection = self.engine.connect()
        try:
            # 修正 select 语法
            result = connection.execute(
                select(self.action_table.c.id, self.action_table.c.action)
                .where(self.action_table.c.label == label)
            ).fetchall()
            # 返回 (id, action) 的元组列表
            return [(row[0], row[1]) for row in result]
        except SQLAlchemyError as e:
            print(f"MYSQL Error occurred: {e} \n occurred in get_actions_by_label")
            return []
        finally:
            connection.close()
