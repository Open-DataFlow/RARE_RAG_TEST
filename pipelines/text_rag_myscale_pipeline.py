from typing import List, Tuple
from multiprocessing import Pool
import torch
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
import os
import glob
from typing import List, Dict, Any
from text_rag_pipeline import RAGPipeline
import multiprocessing as mp
import time
import uuid
import json
from clickhouse_connect import get_client
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from langchain_community.document_loaders import JSONLoader


class MyScaleRAGPipeline(RAGPipeline):
    def __init__(self, config_path: str = "../configs/text_rag_demo.yaml"):
        super().__init__(config_path)
        self.num_gpus = torch.cuda.device_count()
        # MyScale 连接配置
        self.myscale_config = {
            "host": self.config['myscale']['host'],
            "port": self.config['myscale']['port'],
            "database": self.config['myscale']['database'],
            "table_name":self.config['myscale']['table_name']
        }
        self.client = self._init_myscale_client()
        result = self.client.query("SELECT version()")
        print(f"MyScaleDB Version: {result.first_row[0]}")
        self.embeddings = None

    def _init_myscale_client(self):
        client = get_client(
            host=self.myscale_config["host"],
            port=self.myscale_config["port"],
            username="default",
            password='',
            database=self.myscale_config["database"]
        )
        return client

    def _table_exists(self) -> bool:
        """检查表是否已存在且有数据"""
        check_sql = f"""
        SELECT 
            count() AS table_exists 
        FROM system.tables 
        WHERE database = '{self.myscale_config['database']}' 
        AND name = '{self.myscale_config['table_name']}'
        """
        exists = self.client.query(check_sql).first_row[0]
        if exists:
            count_sql = f"SELECT count() FROM {self.myscale_config['database']}.{self.myscale_config['table_name']}"
            row_count = self.client.query(count_sql).first_row[0]
            return row_count > 0
        return False
    def _create_myscale_table(self, embedding_dim: int):
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.myscale_config['database']}.{self.myscale_config['table_name']} (
            id String,
            text String,
            embedding Array(Float32),
            metadata String,
            CONSTRAINT embedding_len CHECK length(embedding) = {embedding_dim}
        ) ENGINE = MergeTree()
        ORDER BY id;
        """
        self.client.command(create_table_sql)  # 修改为command方法
        print("MyScale 表已创建/已存在")

    def _insert_batch_to_myscale(self, batch: List[Tuple[str, str, List[float], str]]):
        if not batch:  # 添加空批次检查
            print("警告：尝试插入空批次")
            return
        data = [
            [
                item[0],  # id
                item[1],  # text
                np.array(item[2], dtype=np.float32).tolist(),  # embedding
                item[3]  # metadata
            ]
            for item in batch
        ]

        # 执行插入操作
        self.client.insert(
            table=f"{self.myscale_config['database']}.{self.myscale_config['table_name']}",
            data=data,
            column_names=["id", "text", "embedding", "metadata"]
        )

    def get_vectorstore(self, embeddings: Any) -> "MyScaleRAGPipeline":
        # 获取嵌入维度
        sample_embedding = embeddings.embed_query("sample")
        embedding_dim = len(sample_embedding)
        #如果存在这个表就不写入
        if self._table_exists():
            print("表已存在且有数据，跳过嵌入计算和写入")
            return self
        # 创建表
        self._create_myscale_table(embedding_dim)
        # 并行处理文档
        all_results = self.load_all_jsonl_documents()
        # 分批插入 MyScale
        batch_size = 10000
        for i in tqdm(range(0, len(all_results), batch_size), desc="插入 MyScale"):
            batch = []
            for text, embedding in all_results[i:i + batch_size]:
                batch.append((
                    str(uuid.uuid4()),  # 生成唯一ID
                    text,
                    embedding,
                    json.dumps({"source": "parallel_processed"})  # 元数据
                ))
            self._insert_batch_to_myscale(batch)

        return self

    def search(self, query: str, k: int = 5) -> List[Dict]:
        query_embedding = self.embeddings.embed_query(query)
        query_embedding_str = str(query_embedding)

        search_sql = f"""
        SELECT 
            id, 
            text, 
            metadata,
            cosineDistance(embedding, {query_embedding_str}) AS cos_dist,
            L2Distance(embedding, {query_embedding_str}) AS l2_dist
        FROM {self.myscale_config['database']}.{self.myscale_config['table_name']}
        ORDER BY {self.config['myscale']['metric_type']}
        LIMIT {k}
        """
        # print(f"执行的SQL: {search_sql}")
        results = []
        for row in self.client.query(search_sql).named_results():
            results.append({
                "id": row['id'],
                "text": row['text'],
                "metadata": json.loads(row['metadata']),
                "cosine_distance": row['cos_dist'],
                "l2_distance": row['l2_dist']
            })

        return results
    @staticmethod
    def process_documents(docs: List[Any], config: Dict) -> List[Any]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['text_splitter']['chunk_size'],
            chunk_overlap=config['text_splitter']['chunk_overlap'],
            add_start_index=True
        )
        return text_splitter.split_documents(docs)
    @staticmethod
    def load_jsonl_documents(file_path: str) -> List[Any]:
        return JSONLoader(
            file_path=file_path,
            jq_schema='.text', #.contents
            text_content=False,
            json_lines=True
        ).load()

    @staticmethod
    def _process_batch_docs(args):
        """静态方法处理文档批次"""
        gpu_id, file_paths, config = args  # 接收config参数
        torch.cuda.set_device(gpu_id)
        embeddings = HuggingFaceEmbeddings(
            model_name=config['model']['embed_model'],
            model_kwargs={'device': f'cuda:{gpu_id}'},
            encode_kwargs={'batch_size': config["model"]["batch_size"]}
        )
        print(f"GPU {gpu_id} 处理 {len(file_paths)} 个文件")
        batch_results = []
        for file_path in file_paths:
            try:
                # 改为调用静态方法，传递config
                docs = MyScaleRAGPipeline.load_jsonl_documents(file_path)
                splits = MyScaleRAGPipeline.process_documents(docs, config)
                print(f"- 文件 {os.path.basename(file_path)} 分割为 {len(splits)} 个片段")
                texts = [doc.page_content for doc in splits]
                text_embeddings = embeddings.embed_documents(texts)
                batch_results.extend(zip(texts, text_embeddings))
            except Exception as e:
                print(f"GPU {gpu_id} 处理文件 {file_path} 出错: {str(e)}")
        return batch_results
    def load_all_jsonl_documents(self) -> List[Any]:
        jsonl_files = glob.glob(os.path.join(self.config['paths']['jsonl_dir'], "*.jsonl"))
        if not jsonl_files:
            raise ValueError(f"目录 {self.config['paths']['jsonl_dir']} 中没有找到JSONL文件")
        print(f"找到 {len(jsonl_files)} 个JSONL文件，使用 {self.num_gpus} 张GPU并行处理")
        start_time = time.time()
        # 均匀分配文件到各个GPU
        file_chunks = [jsonl_files[i::self.num_gpus] for i in range(self.num_gpus)]
        chunk_sizes = [len(chunk) for chunk in file_chunks]

        args = [(i, chunk, self.config) for i, chunk in enumerate(file_chunks)]

        results = []
        with Pool(processes=self.num_gpus) as pool:
            with tqdm(total=len(jsonl_files), desc="处理文件") as pbar:
                # 修改为传递args
                for i, batch_result in enumerate(pool.imap_unordered(
                        self._process_batch_docs,
                        args
                )):
                    results.extend(batch_result)
                    pbar.update(chunk_sizes[i])

        total_time = time.time() - start_time
        print(f"并行处理完成，总耗时: {total_time:.2f}秒")
        print(f"处理速度: {len(jsonl_files) / total_time:.2f} 文件/秒")
        return results

    def run(self):
        total_start_time = time.time()

        # 初始化嵌入模型
        start_time = time.time()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config['model']['embed_model'],
            encode_kwargs={'batch_size': 128}
        )
        embed_init_time = time.time() - start_time
        print(f"\n[耗时] 嵌入模型初始化: {embed_init_time:.2f}秒")

        # 检查表是否已存在且有数据
        if self._table_exists():
            print("检测到表已存在且有数据，跳过文档处理和嵌入写入")
        else:
            start_time = time.time()
            self.get_vectorstore(self.embeddings)
            vectorstore_time = time.time() - start_time
            print(f"[耗时] 向量数据库加载/创建: {vectorstore_time:.2f}秒")

        # 处理每个查询
        for query in self.config['queries']:
            query_start_time = time.time()
            print(f"\n{'=' * 50}")
            print(f"原始问题: {query}")

            # 1. 在 MyScale 中检索相关文档
            start_time = time.time()
            results = self.search(query, k=self.config['retriever']['k'])
            # print(f"检索信息：{results}")
            retrieval_time = time.time() - start_time
            print(f"[耗时] 检索阶段: {retrieval_time:.2f}秒")
            if not results:
                print("未找到相关结果")
                continue

            # 2. 打印检索结果
            print("\n检索到的相关内容:")
            for i, doc in enumerate(results, 1):
                print(f"[结果 {i}] (距离: {doc['cosine_distance']:.6f})")
                print(doc['text'][:200] + "...")
            context = "\n".join([f"【文档片段 {i}】\n{doc['text']}" for i, doc in enumerate(results, 1)])
            print("\n检索结果整合:")
            print(context)
            # 单次查询总耗时
            query_total_time = time.time() - query_start_time
            print(f"[总耗时] 当前查询处理: {query_total_time:.2f}秒")

        # 整体流程总耗时
        total_time = time.time() - total_start_time
        print(f"\n{'=' * 50}")
        print(f"[总耗时] 整个RAG Pipeline: {total_time:.2f}秒")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    pipeline = MyScaleRAGPipeline()
    pipeline.run()