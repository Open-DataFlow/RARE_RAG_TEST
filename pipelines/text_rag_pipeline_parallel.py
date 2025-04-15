from multiprocessing import Pool
import torch
from tqdm import tqdm
import math
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import yaml
import http.client
import json
import glob
from typing import List, Dict, Any
from text_rag_pipeline import RAGPipeline
import multiprocessing as mp

# if __name__ == '__main__':
#     mp.set_start_method('spawn')

class ParallelRAGPipeline(RAGPipeline):
    def __init__(self, config_path: str = "../configs/text_rag_demo.yaml"):
        super().__init__(config_path)
        self.num_gpus = torch.cuda.device_count()

    def _process_single_file(self, file_args):
        """处理单个文件的函数，适配多进程调用"""
        file_path, gpu_id = file_args
        torch.cuda.set_device(gpu_id)

        try:
            # 加载并处理单个文件
            docs = self.load_jsonl_documents(file_path)
            splits = self.process_documents(docs)

            # 初始化当前GPU上的embedding模型
            embeddings = HuggingFaceEmbeddings(
                model_name=self.config['model']['embed_model'],
                model_kwargs={'device': f'cuda:{gpu_id}'}
            )

            # 为当前文件的chunks生成embeddings
            texts = [doc.page_content for doc in splits]
            embeddings.embed_documents(texts)

            return splits
        except Exception as e:
            print(f"Error processing {file_path} on GPU {gpu_id}: {str(e)}")
            return []

    def load_all_jsonl_documents(self) -> List[Any]:
        """并行加载和处理所有JSONL文件"""
        jsonl_files = glob.glob(os.path.join(self.config['paths']['jsonl_dir'], "*.jsonl"))
        if not jsonl_files:
            raise ValueError(f"目录 {self.config['paths']['jsonl_dir']} 中没有找到JSONL文件")

        print(f"找到 {len(jsonl_files)} 个JSONL文件，使用 {self.num_gpus} 张GPU并行处理")

        # 分配文件到不同的GPU
        gpu_assignments = []
        for i, file_path in enumerate(jsonl_files):
            gpu_id = i % self.num_gpus
            gpu_assignments.append((file_path, gpu_id))

        # 使用多进程并行处理
        results = []
        with Pool(processes=self.num_gpus) as pool:
            with tqdm(total=len(jsonl_files), desc="处理JSONL文件") as pbar:
                for batch_result in pool.imap_unordered(self._process_single_file, gpu_assignments):
                    results.extend(batch_result)
                    pbar.update(1)

        print(f"共处理 {len(results)} 个文档片段")
        return results

    def get_vectorstore(self, embeddings: Any) -> Chroma:
        """修改后的向量数据库初始化方法"""
        if os.path.exists(self.config['paths']['persist_dir']):
            print("检测到已有向量数据库，直接加载...")
            return Chroma(
                persist_directory=self.config['paths']['persist_dir'],
                embedding_function=embeddings
            )
        else:
            print("创建新向量数据库...")
            # 使用并行加载
            splits = self.load_all_jsonl_documents()

            # 统计信息
            total_chunks = len(splits)
            avg_chunk_len = sum(len(d.page_content) for d in splits) / total_chunks
            print(f"\n知识库统计:")
            print(f"- 处理后片段数: {total_chunks}")
            print(f"- 平均片段长度: {avg_chunk_len:.0f}字符")

            # 在主GPU上创建向量库
            main_device = f"cuda:{0}"
            with torch.cuda.device(main_device):
                embeddings.model_kwargs['device'] = main_device
                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    persist_directory=self.config['paths']['persist_dir']
                )
            return vectorstore
if __name__ == "__main__":
    pipeline = ParallelRAGPipeline()
    pipeline.run()