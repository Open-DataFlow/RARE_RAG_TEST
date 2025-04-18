**配置文件：**

```yaml
myscale:
  host: "localhost"
  port: 8123
  table_name: "mp_test_vector" #数据库表名称
  database: "default"
  metric_type: "cos_dist"  # l2_dist

# 路径配置 Chroma
paths:
  # JSONL文件存放目录（支持多个文件）
  jsonl_dir: "../data/raw/text/" #需要处理的数据路径
  # 向量数据库持久化存储目录
  persist_dir: "../outputs/bge_m3_chroma_db_4_gpu_mp_ch1024" #"./chroma_db"

  collection_name: "rag_collection" #Chroma

# 模型配置
model:
  # 嵌入模型本地路径（需提前从https://hf-mirror.com/BAAI/bge-large-en-v1.5下载）
#  "/home/liuzhou/projects/Rare_rag/bge-large-en-v1.5"
  embed_model: "/mnt/public/model/bge-m3"
  # 聊天模型名称
  chat_model: "gpt-3.5-turbo" #暂时用不到

  batch_size: 128

# 文本分割器配置
text_splitter:
  # 每个文本块的最大长度
  chunk_size: 2048
  # 相邻文本块的重叠长度
  chunk_overlap: 400

# 检索器配置
retriever:
  # 每次检索返回的最大结果数量
  k: 5

# API接口配置
api:
  # ChatAnywhere服务端点
  endpoint: "api.chatanywhere.tech"
  # API请求路径
  path: "/v1/chat/completions"
  # 身份验证密钥（示例密钥，需替换为实际值）
  api_key: "sk-ffSAwSlifFVkCujrLcUDfamzpVgjliaFJ4zA4sEv3hT6DNe9"
  # 系统提示词
  system_prompt: "You are a helpful assistant that answers questions based on the provided context."

# 测试查询列表
queries:
  - "Jupiter's Water"       # 木星水资源
```

**使用案例：**

**并行+Myscale：**

```
python text_rag_myscale_pipeline.py
```

**并行+Chroma：**

```
python text_rag_pipeline_parallel
```

---

