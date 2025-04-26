import sqlite3
import numpy as np
import faiss

class VoiceprintStorage:
    def __init__(self, sqlite_db_path='users.db', faiss_index_path_189='faiss_index_189.index', faiss_index_path_512='faiss_index_512.index'):
        self.sqlite_db_path = sqlite_db_path
        self.faiss_index_path_189 = faiss_index_path_189
        self.faiss_index_path_512 = faiss_index_path_512
        self.conn = None
        self.cursor = None
        self.index_189 = None
        self.index_512 = None
        self._initialize()

    def _initialize(self):
        self._initialize_sqlite()
        self._initialize_faiss()

    def _initialize_sqlite(self):
        self.conn = sqlite3.connect(self.sqlite_db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            other_info TEXT,
            index_id_189 INTEGER,
            index_id_512 INTEGER
        )
        ''')
        self.conn.commit()

    def _initialize_faiss(self):
        # 尝试加载已存在的 Faiss 索引
        try:
            self.index_189 = faiss.read_index(self.faiss_index_path_189)
            self.index_512 = faiss.read_index(self.faiss_index_path_512)
        except:
            # 如果索引文件不存在，则创建新的索引并立即保存到磁盘
            self.index_189 = faiss.IndexFlatL2(189)
            self.index_512 = faiss.IndexFlatL2(512)
            self._save_faiss_indices()

    def add_voiceprint(self, embedding_189, embedding_512):
        # 将向量转换为 Faiss 所需的格式
        embedding_189 = np.array([embedding_189], dtype=np.float32)
        embedding_512 = np.array([embedding_512], dtype=np.float32)
        
        # 添加向量到索引
        index_id_189 = self.index_189.ntotal
        index_id_512 = self.index_512.ntotal
        
        self.index_189.add(embedding_189)
        self.index_512.add(embedding_512)

        # 保存索引到磁盘
        self._save_faiss_indices()

        return index_id_189, index_id_512

    def add_user(self, name, other_info=None, index_id_189=None, index_id_512=None):
        self.cursor.execute('INSERT INTO users (name, other_info, index_id_189, index_id_512) VALUES (?, ?, ?, ?)', (name, other_info, index_id_189, index_id_512))
        self.conn.commit()
        return self.cursor.lastrowid

    def get_user_by_name(self, name):
        self.cursor.execute('SELECT * FROM users WHERE name = ?', (name,))
        return self.cursor.fetchone()

    def get_user_by_id(self, user_id):
        self.cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        return self.cursor.fetchone()

    def get_user_by_index_id_189(self, index_id_189):
        self.cursor.execute('SELECT * FROM users WHERE index_id_189 = ?', (index_id_189,))
        return self.cursor.fetchone()

    def get_user_by_index_id_512(self, index_id_512):
        self.cursor.execute('SELECT * FROM users WHERE index_id_512 = ?', (index_id_512,))
        return self.cursor.fetchone()

    def search_voiceprint_189(self, embedding_189, limit=1):
        # 将查询向量转换为 Faiss 所需的格式
        embedding_189 = np.array([embedding_189], dtype=np.float32)
        # 搜索最近的向量
        distances, indices = self.index_189.search(embedding_189, limit)
        # 获取最相似的向量数据
        similar_vectors = self.index_189.reconstruct_batch(indices[0])
        return distances, indices, similar_vectors

    def search_voiceprint_512(self, embedding_512, limit=1):
        # 将查询向量转换为 Faiss 所需的格式
        embedding_512 = np.array([embedding_512], dtype=np.float32)
        # 搜索最近的向量
        distances, indices = self.index_512.search(embedding_512, limit)
        # 获取最相似的向量数据
        similar_vectors = self.index_512.reconstruct_batch(indices[0])
        return distances, indices, similar_vectors

    def _save_faiss_indices(self):
        # 保存 Faiss 索引到磁盘
        faiss.write_index(self.index_189, self.faiss_index_path_189)
        faiss.write_index(self.index_512, self.faiss_index_path_512)

    def close(self):
        # 保存索引到磁盘
        self._save_faiss_indices()
        # 关闭数据库连接
        self.conn.close()

    def delete_user(self, user_id):
        self.cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))
        self.conn.commit()

    def get_all_users(self):
        self.cursor.execute('SELECT * FROM users')
        return self.cursor.fetchall()

    def save_audio_info(self, speaker_name, audio_path):
        """
        Save the audio file path associated with a speaker locally.
        
        Args:
            speaker_name (str): Name of the speaker.
            audio_path (str): Path to the audio file.
        """
        import json
        import os
        
        # Define the local file path
        local_file = "audio_info.json"
        data = {}
        
        # Load existing data if file exists
        if os.path.exists(local_file):
            with open(local_file, 'r') as f:
                data = json.load(f)
        
        # Update data with new entry
        if speaker_name not in data:
            data[speaker_name] = []
        data[speaker_name].append(audio_path)
        
        # Save data back to file
        with open(local_file, 'w') as f:
            json.dump(data, f, indent=4)

    def get_audio_info(self, speaker_name):
        """
        Retrieve the saved audio file paths for a speaker.
        
        Args:
            speaker_name (str): Name of the speaker.
        
        Returns:
            list: List of saved audio file paths, or empty list if no data exists.
        """
        import json
        import os
        
        local_file = "audio_info.json"
        if not os.path.exists(local_file):
            return []
        
        with open(local_file, 'r') as f:
            data = json.load(f)
        
        return data.get(speaker_name, [])
