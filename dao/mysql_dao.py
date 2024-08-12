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
                                Column('voiceprint', Integer),
                                Column('permission_level', Integer))

        self.action_table = Table('action', self.metadata,
                                  Column('id', Integer, primary_key=True),
                                  Column('action', String(255), unique=True))

    def create_mysql_table(self, table_name):
        if table_name == 'user':
            self.user_table.create(self.engine, checkfirst=True)
        elif table_name == 'action':
            self.action_table.create(self.engine, checkfirst=True)

    def load_data_to_mysql(self, table_name, data):
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
                for entry in data
            ]
            connection.execute(table.insert(), formatted_data)
            connection.commit()
        except SQLAlchemyError as e:
            connection.rollback()
            print(f"Error occurred: {e}")
        finally:
            connection.close()

    def create_action_table(self):
        self.create_mysql_table('action')

    def insert_action(self, action):
        connection = self.engine.connect()
        try:
            connection.execute(self.action_table.insert(), {'action': action})
            connection.commit()
        except SQLAlchemyError as e:
            connection.rollback()
            print(f"Error occurred: {e}")
        finally:
            connection.close()

    def delete_action(self, action):
        connection = self.engine.connect()
        try:
            connection.execute(delete(self.action_table).where(self.action_table.c.action == action))
            connection.commit()
        except SQLAlchemyError as e:
            connection.rollback()
            print(f"Error occurred: {e}")
        finally:
            connection.close()

    def get_all_actions(self):
        connection = self.engine.connect()
        try:
            result = connection.execute(select(self.action_table.c.action)).fetchall()
            return [row[0] for row in result]
        except SQLAlchemyError as e:
            print(f"Error occurred: {e}")
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
            print(f"Error occurred: {e}")
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
            print(f"Error occurred: {e}")
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
            print(f"Error occurred: {e}")
            return []
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
            print(f"Error occurred: {e}")
        finally:
            connection.close()
