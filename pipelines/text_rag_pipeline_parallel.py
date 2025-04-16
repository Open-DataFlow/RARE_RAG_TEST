from multiprocessing import Pool
import torch
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import glob
from typing import List, Dict, Any
from text_rag_pipeline import RAGPipeline
import multiprocessing as mp
import time
import uuid
import chromadb
import chromadb.config

class ParallelRAGPipeline(RAGPipeline):
    def __init__(self, config_path: str = "../configs/text_rag_demo.yaml"):
        super().__init__(config_path)
        self.num_gpus = torch.cuda.device_count()

    def _process_batch(self, args):
        """处理一批文件"""
        gpu_id, file_paths = args
        torch.cuda.set_device(gpu_id)
        # 初始化模型
        embeddings = HuggingFaceEmbeddings(
            model_name=self.config['model']['embed_model'],
            model_kwargs={'device': f'cuda:{gpu_id}'},
            encode_kwargs={'batch_size': self.config["model"]["batch_size"]}
        )
        print(args)

        batch_results = []
        for file_path in file_paths:
            try:
                docs = self.load_jsonl_documents(file_path)
                splits = self.process_documents(docs)
                print(f"- 处理后片段数: {len(splits)}")
                batch_results.extend([
                    (doc.page_content, embeddings.embed_query(doc.page_content))
                    for doc in splits
                ])
            except Exception as e:
                print(f"Error on GPU {gpu_id}: {str(e)}")
        print(len(batch_results))
        return batch_results

    def _process_batch_docs(self, args):
        """处理一批文件（使用embed_documents批量处理）"""
        gpu_id, file_paths = args
        torch.cuda.set_device(gpu_id)
        embeddings = HuggingFaceEmbeddings(
            model_name=self.config['model']['embed_model'],
            model_kwargs={'device': f'cuda:{gpu_id}'},
            encode_kwargs={'batch_size': self.config["model"]["batch_size"]}
        )
        print(f"GPU {gpu_id} 处理 {len(file_paths)} 个文件")

        batch_results = []
        for file_path in file_paths:
            try:
                docs = self.load_jsonl_documents(file_path)
                splits = self.process_documents(docs)
                print(f"- 文件 {os.path.basename(file_path)} 分割为 {len(splits)} 个片段")
                texts = [doc.page_content for doc in splits]
                text_embeddings = embeddings.embed_documents(texts)
                batch_results.extend(zip(texts, text_embeddings))
            except Exception as e:
                print(f"GPU {gpu_id} 处理文件 {file_path} 出错: {str(e)}")

        print(f"GPU {gpu_id} 完成，共生成 {len(batch_results)} 个嵌入向量")
        return batch_results
    def load_all_jsonl_documents(self) -> List[Any]:
        jsonl_files = glob.glob(os.path.join(self.config['paths']['jsonl_dir'], "*.jsonl"))
        if not jsonl_files:
            raise ValueError(f"目录 {self.config['paths']['jsonl_dir']} 中没有找到JSONL文件")
        print(f"找到 {len(jsonl_files)} 个JSONL文件，使用 {self.num_gpus} 张GPU并行处理")
        start_time = time.time()
        # 均匀分配文件到各个GPU
        file_chunks = [jsonl_files[i::self.num_gpus] for i in range(self.num_gpus)]

        with Pool(processes=self.num_gpus) as pool:
            results = []
            with tqdm(total=len(jsonl_files), desc="处理文件") as pbar:
                for batch_result in pool.imap(
                        # self._process_batch, 单个chunk
                        self._process_batch_docs,
                        enumerate(file_chunks)
                ):
                    results.extend(batch_result)
                    print(len(results))
                    pbar.update(len(file_chunks[0]))  # 假设均匀分配

        total_time = time.time() - start_time
        print(f"并行处理完成，总耗时: {total_time:.2f}秒")
        print(f"处理速度: {len(jsonl_files) / total_time:.2f} 文件/秒")

        return results

    def get_vectorstore(self, embeddings: Any) -> Chroma:
        """使用并行计算的embeddings创建向量库"""
        persist_dir = self.config['paths']['persist_dir']
        collection_name = self.config['paths']['collection_name']

        if os.path.exists(persist_dir):
            print("加载已有向量库...")
            return Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )

        print("创建新向量数据库(并行处理)...")
        all_results = self.load_all_jsonl_documents()

        # 准备数据
        documents = []
        embeddings_list = []
        metadatas = []

        for text, embedding in all_results:
            documents.append(text)
            embeddings_list.append(embedding)
            metadatas.append({"source": "processed"})
        with torch.cuda.device(0):
            chroma_client = chromadb.PersistentClient(path=persist_dir)
            collection = chroma_client.get_or_create_collection(name=collection_name)
            batch_size = 10000
            for i in tqdm(range(0, len(documents), batch_size)):
                batch_ids = [str(uuid.uuid4()) for _ in documents[i:i + batch_size]]
                collection.upsert(
                    ids=batch_ids,
                    embeddings=embeddings_list[i:i + batch_size],
                    documents=documents[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size]
                )
            return Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_dir,
                client=chroma_client
            )
if __name__ == "__main__":
    mp.set_start_method('spawn')
    pipeline = ParallelRAGPipeline()
    pipeline.run()