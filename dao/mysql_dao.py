import pymysql
from pymysql.err import OperationalError


# Mysql数据库
class MySQLClient:
    def __init__(self, host, port, user, password, database):
        self.cursor = None
        self.connection = None
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connect()

    def connect(self):
        try:
            self.connection = pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password,
                                              database=None)
            self.cursor = self.connection.cursor()
            self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            self.connection.select_db(self.database)
        except OperationalError as e:
            print(f"Error connecting to MySQL: {e}")
            raise

    def create_mysql_table(self, table_name):
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255),
                voiceprint BIGINT,
                permission_level SMALLINT
            );
            """
        self.cursor.execute(create_table_sql)
        self.connection.commit()

    def load_data_to_mysql(self, table_name, data):
        insert_sql = f"INSERT INTO {table_name} (username, voiceprint, permission_level) VALUES (%s, %s, %s)"
        self.cursor.executemany(insert_sql, data)
        self.connection.commit()
        user_id = self.cursor.lastrowid
        return user_id

    def create_action_table(self):
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS action (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            action VARCHAR(255) UNIQUE
        );
        """
        self.cursor.execute(create_table_sql)
        self.connection.commit()

    def insert_action(self, action):
        insert_sql = f"INSERT INTO action (action) VALUES (%s)"
        self.cursor.execute(insert_sql, (action,))
        self.connection.commit()

    def delete_action(self, action):
        delete_sql = f"DELETE FROM action WHERE action = %s"
        self.cursor.execute(delete_sql, (action,))
        self.connection.commit()

    def get_all_actions(self):
        select_sql = "SELECT action FROM action"
        self.cursor.execute(select_sql)
        results = self.cursor.fetchall()
        return [result[0] for result in results]

    def get_action_id(self, action):
        select_sql = f"SELECT id FROM action WHERE action = %s"
        self.cursor.execute(select_sql, (action,))
        result = self.cursor.fetchone()
        ans = result[0] or None
        return ans
