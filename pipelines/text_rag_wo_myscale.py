from typing import List, Tuple, Dict, Any
from multiprocessing import Pool
import torch
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
import os
import glob
import time
import uuid
import json
from text_rag_pipeline import RAGPipeline
import multiprocessing as mp

class MyScaleRAGPipeline(RAGPipeline):
    def __init__(self, config_path: str = "../configs/text_rag_demo.yaml"):
        super().__init__(config_path)
        self.num_gpus = torch.cuda.device_count()
        self.embeddings = None

    def _save_embeddings_to_file(self, results: List[Tuple[str, list]], save_path: str = "embeddings.jsonl"):
        """将所有embedding结果保存为每行一个JSON的文件"""
        with open(save_path, "w", encoding="utf-8") as f:
            for text, embedding in results:
                item = {
                    "id": str(uuid.uuid4()),
                    # "text": text,
                    "embedding": embedding
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Embedding结果已保存到: {save_path}")

    def get_vectorstore(self, embeddings: Any, save_path: str = "embeddings.jsonl") -> "MyScaleRAGPipeline":
        # 获取嵌入维度（可选）
        sample_embedding = embeddings.embed_query("sample")
        embedding_dim = len(sample_embedding)
        print(f"Embedding维度为: {embedding_dim}")

        # 并行处理文档
        all_results = self.load_all_jsonl_documents()
        print(f"共生成 {len(all_results)} 条 embedding 结果")

        # 保存到本地文件
        self._save_embeddings_to_file(all_results, save_path)
        return self

    def _process_batch_docs(self, args):
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
        chunk_sizes = [len(chunk) for chunk in file_chunks]

        results = []
        with Pool(processes=self.num_gpus) as pool:
            with tqdm(total=len(jsonl_files), desc="处理文件") as pbar:
                for i, batch_result in enumerate(pool.imap_unordered(
                        self._process_batch_docs,
                        enumerate(file_chunks)
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

        # 只生成embedding并保存，不再入库
        start_time = time.time()
        self.get_vectorstore(self.embeddings)
        vectorstore_time = time.time() - start_time
        print(f"[耗时] 向量文件保存: {vectorstore_time:.2f}秒")

        total_time = time.time() - total_start_time
        print(f"\n{'=' * 50}")
        print(f"[总耗时] 整个Embedding Pipeline: {total_time:.2f}秒")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    pipeline = MyScaleRAGPipeline()
    pipeline.run()