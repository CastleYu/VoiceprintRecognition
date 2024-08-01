import pymysql
from pymysql.connections import Connection
from pymysql.err import OperationalError


class MySQLClient:
    def __init__(self, host, port, user, password, database):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database

    def get_connection(self) -> Connection:
        try:
            connection = pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password,
                                         database=self.database)
            return connection
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
        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.execute(create_table_sql)
        connection.commit()
        cursor.close()
        connection.close()

    def load_data_to_mysql(self, table_name, data):
        insert_sql = f"INSERT INTO {table_name} (username, voiceprint, permission_level) VALUES (%s, %s, %s)"
        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.executemany(insert_sql, data)
        connection.commit()
        user_id = cursor.lastrowid
        cursor.close()
        connection.close()
        return user_id

    def create_action_table(self):
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS action (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            action VARCHAR(255) UNIQUE
        );
        """
        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.execute(create_table_sql)
        connection.commit()
        cursor.close()
        connection.close()

    def insert_action(self, action):
        insert_sql = f"INSERT INTO action (action) VALUES (%s)"
        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.execute(insert_sql, (action,))
        connection.commit()
        cursor.close()
        connection.close()

    def delete_action(self, action):
        delete_sql = f"DELETE FROM action WHERE action = %s"
        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.execute(delete_sql, (action,))
        connection.commit()
        cursor.close()
        connection.close()

    def get_all_actions(self):
        select_sql = "SELECT action FROM action"
        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.execute(select_sql)
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        return [result[0] for result in results]

    def get_action_id(self, action):
        select_sql = f"SELECT id FROM action WHERE action = %s"
        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.execute(select_sql, (action,))
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        ans = result[0] or None
        return ans

    def delete_user(self, user_id):
        delete_sql = f"DELETE FROM user WHERE id = %s"
        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.execute(delete_sql, (user_id,))
        connection.commit()

    def get_all_users(self):
        select_sql = "SELECT * FROM user"
        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.execute(select_sql)
        results = cursor.fetchall()

        users = []
        for result in results:
            user = {
                "id": result[0],
                "username": result[1],
                "voiceprint": result[2],
                "permission_level": result[3]
            }
            users.append(user)

        return users

    def update_user_info(self, table_name, user_id, username=None, permission_level=None):
        update_fields = []
        update_values = []

        if username is not None:
            update_fields.append("username = %s")
            update_values.append(username)

        if permission_level is not None:
            update_fields.append("permission_level = %s")
            update_values.append(permission_level)

        if not update_fields:
            return

        update_sql = f"UPDATE {table_name} SET {', '.join(update_fields)} WHERE id = %s"
        update_values.append(user_id)
        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.execute(update_sql, tuple(update_values))
        connection.commit()

        return
