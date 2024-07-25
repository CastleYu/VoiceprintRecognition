## 1. /load

### 描述
上传音频文件并获取其特征向量，将特征向量插入 Milvus 并存储到 MySQL。

### 方法
PUT

### 请求参数
- 文件：通过 `multipart/form-data` 上传的音频文件。

### 响应
- 成功：
  ```json
  {
    "result": "SUCCESS",
    "data": {
      "user_id": 0,
      "milvus_ids": [<milvus_id>],
      "voiceprint": [<milvus_id>]
    }
  }
  ```
- 失败：
  ```json
  {
    "result": "FAILED",
    "data": {
      "error": "<错误信息>"
    }
  }
  ```

## 2. /asr

### 描述
上传音频文件并进行语音识别，返回识别出的文本。

### 方法
POST

### 请求参数
- 文件：通过 `multipart/form-data` 上传的音频文件。

### 响应
- 成功：
  ```json
  {
    "result": "SUCCESS",
    "data": {
      "text": "<识别出的文本>"
    }
  }
  ```
- 失败：
  ```json
  {
    "result": "FAILED",
    "data": {
      "error": "<错误信息>"
    }
  }
  ```

## 3. /recognize

### 描述
上传音频文件进行识别，返回与数据库中最相似的音频信息及识别结果。

### 方法
POST

### 请求参数
- 文件：通过 `multipart/form-data` 上传的音频文件。

### 响应
- 成功：
  ```json
  {
    "result": "SUCCESS",
    "data": {
      "user_id": <用户ID>,
      "similar_distance": <相似度距离>,
      "similarity_score": <相似度评分>,
      "asr_result": "<识别出的文本>"
    }
  }
  ```
- 失败：
  ```json
  {
    "result": "FAILED",
    "data": {
      "error": "<错误信息>"
    }
  }
  ```

## 4. /add_action

### 描述
添加一个新的指令到 MySQL。

### 方法
POST

### 请求参数
- `action`：要添加的指令内容，使用 `application/x-www-form-urlencoded` 进行传输。

### 响应
- 成功：
  ```json
  {
    "result": "SUCCESS"
  }
  ```

## 5. /delete_action

### 描述
删除指定的指令。

### 方法
POST

### 请求参数
- `action`：要删除的指令内容，使用 `application/x-www-form-urlencoded` 进行传输。

### 响应
- 成功：
  ```json
  {
    "result": "SUCCESS"
  }
  ```

## 6. /search_action

### 描述
根据关键词搜索最匹配的指令。

### 方法
GET

### 请求参数
- `action`：要搜索的关键词，使用 `application/x-www-form-urlencoded` 进行传输。

### 响应
- 成功：
  ```json
  {
    "result": "SUCCESS",
    "data": {
      "action_id": <指令ID>,
      "best_match_action": "<最匹配的指令>",
      "similarity_percent": "<相似度百分比>"
    }
  }
  ```