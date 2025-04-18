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
import sys
import re
import tempfile
import shutil
from filelock import FileLock
from transformers import AutoTokenizer
class MyScaleRAGPipeline(RAGPipeline):
    def __init__(self, config_path: str = "../configs/text_rag_demo.yaml"):
        super().__init__(config_path)
        self.num_gpus = torch.cuda.device_count()
        self.embeddings = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def analyze_chunk_token_stats(self, splits: List[Any]):
        model_name = self.config['model']['embed_model']
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        token_counts = [len(tokenizer.encode(doc.page_content, truncation=False)) for doc in splits]

        total_chunks = len(token_counts)
        max_tokens = max(token_counts)
        min_tokens = min(token_counts)
        avg_tokens = sum(token_counts) / total_chunks

        print(f"\n[Chunk Token统计]")
        print(f"- 总chunk数: {total_chunks}")
        print(f"- 最大token数: {max_tokens}")
        print(f"- 最小token数: {min_tokens}")
        print(f"- 平均token数: {avg_tokens:.1f}")

        return token_counts

    def estimate_embedding_memory(self, token_counts: list, batch_size: int, embedding_dim: int, dtype_bytes: int = 2):
        sorted_tokens = sorted(token_counts, reverse=True)
        max_batch_tokens = max(sorted_tokens[:batch_size]) if len(sorted_tokens) >= batch_size else max(sorted_tokens)
        base_mem = batch_size * max_batch_tokens * embedding_dim * dtype_bytes / (1024 ** 2)  # MB
        recommend_mem = base_mem * 2.5 / 1024  # GB

        print(f"\n[显存估算]")
        print(f"- 假定 batch_size = {batch_size}, 最大token数 = {max_batch_tokens}")
        print(f"- embedding维度 = {embedding_dim}")
        print(f"- 理论最小显存: {base_mem / 1024:.2f} GB")
        print(f"- 推荐显存（含缓存）：{recommend_mem:.2f} GB")

        return recommend_mem

    def _save_embeddings_to_file(self, results: List[Tuple[str, list]], save_path: str = "embeddings.jsonl"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for text, embedding in results:
                item = {
                    "id": str(uuid.uuid4()),
                    "text": text,
                    "embedding": embedding
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Embedding结果已保存到: {save_path}")

    def get_vectorstore(self, embeddings: Any, save_path: str = "embeddings.jsonl") -> "MyScaleRAGPipeline":
        sample_embedding = embeddings.embed_query("sample")
        embedding_dim = len(sample_embedding)
        print(f"Embedding维度为: {embedding_dim}")
        all_results = self.load_all_jsonl_documents()
        # print(f"共生成 {len(all_results)} 条 embedding 结果")
        self._save_embeddings_to_file(all_results, save_path)
        return self
    def _process_batch_docs(self, args):
        gpu_id, file_paths = args
        torch.cuda.set_device(gpu_id)

        # 每个进程写入独立临时文件
        per_gpu_file = f"{self.output_file}.gpu{gpu_id}.jsonl"
        os.makedirs(os.path.dirname(per_gpu_file), exist_ok=True)

        embeddings = HuggingFaceEmbeddings(
            model_name=self.config['model']['embed_model'],
            model_kwargs={'device': f'cuda:{gpu_id}'},
            encode_kwargs={'batch_size': self.config["model"]["batch_size"]}
        )

        with open(per_gpu_file, 'w', encoding='utf-8') as f:
            for file_path in tqdm(file_paths, desc=f"GPU {gpu_id} 文件进度"):
                current_file = os.path.basename(file_path)
                try:
                    docs = self.load_jsonl_documents(file_path)
                    splits = self.process_documents(docs)
                    texts = [doc.page_content for doc in splits]
                    token_counts = self.analyze_chunk_token_stats(splits)
                    recommend_mem = self.estimate_embedding_memory(token_counts, 128, 1024)
                    if recommend_mem > 64:
                        print(f"\n 文件 {current_file} 估算显存需求高达 {recommend_mem:.2f} GB，已超64GB！请重点关注。")

                    text_embeddings = []
                    for i in range(0, len(texts), self.config["model"]["batch_size"]):
                        batch = texts[i:i + self.config["model"]["batch_size"]]
                        try:
                            with torch.no_grad():
                                pass
                                # batch_embeddings = embeddings.embed_documents(batch)
                        except RuntimeError as e:
                            if "CUDA out of memory" in str(e):
                                print(f"\n❌ GPU {gpu_id} 显存不足! "
                                      f"当前文件: {current_file} | "
                                      f"batch范围: {i}-{i + self.config['model']['batch_size']}\n"
                                      f"建议: 降低batch_size或减小文本长度")
                                raise
                            raise
                        # text_embeddings.extend(batch_embeddings)
                        # if i % 100 == 0:
                        #     torch.cuda.empty_cache()

                    # for text, embedding in zip(texts, text_embeddings):
                    #     item = {
                    #         "id": str(uuid.uuid4()),
                    #         "text": text,
                    #         "embedding": embedding
                    #     }
                    #     f.write(json.dumps(item, ensure_ascii=False) + '\n')

                except Exception as e:
                    print(f"GPU {gpu_id} 处理文件 {file_path} 出错: {str(e)}")
                    continue

        return gpu_id
    def _merge_temp_files(self):
        with open(self.output_file, 'w', encoding='utf-8') as out_f:
            for gid in range(self.num_gpus):
                tmp_file = f"{self.output_file}.gpu{gid}.jsonl"
                try:
                    with open(tmp_file, 'r', encoding='utf-8') as in_f:
                        pass
                        # out_f.write(in_f.read())
                    # os.remove(tmp_file)
                except FileNotFoundError:
                    print(f"警告: 临时文件 {tmp_file} 不存在")
    def _get_file_index(self, filename: str) -> int:
        match = re.search(r'2048ch_cleaned_en_best_(\d+)\.jsonl', filename)
        if match:
            return int(match.group(1))
        return -1
    @staticmethod
    def _init_child_process(lock_path):
        global process_lock
        process_lock = FileLock(lock_path)
        torch.cuda.empty_cache()
    def load_all_jsonl_documents(self) -> List[Any]:
        jsonl_files = glob.glob(os.path.join(self.config['paths']['jsonl_dir'], "2048ch_cleaned_en_best_*.jsonl"))
        if not hasattr(self, 'start_idx') or not hasattr(self, 'end_idx'):
            filtered_files = jsonl_files
        else:
            filtered_files = [
                f for f in jsonl_files
                if self.start_idx <= self._get_file_index(os.path.basename(f)) <= self.end_idx
            ]

        if not filtered_files:
            raise ValueError(f"目录 {self.config['paths']['jsonl_dir']} 中没有找到符合条件的JSONL文件")

        filtered_files.sort(key=lambda x: self._get_file_index(os.path.basename(x)))
        print(f"找到 {len(filtered_files)} 个JSONL文件，使用 {self.num_gpus} 张GPU并行处理")

        # 2. 创建临时工作区
        temp_dir = tempfile.mkdtemp(prefix="rag_processing_")
        lock_file = os.path.join(temp_dir, "merge.lock")

        # 4. 多进程处理
        start_time = time.time()
        file_chunks = [filtered_files[i::self.num_gpus] for i in range(self.num_gpus)]
        chunk_sizes = [len(chunk) for chunk in file_chunks]

        with Pool(
                processes=self.num_gpus,
                initializer=self._init_child_process,
                initargs=(lock_file,)
        ) as pool:
            with tqdm(total=len(filtered_files), desc="处理进度") as pbar:
                for gpu_id in pool.imap_unordered(
                        self._process_batch_docs,
                        enumerate(file_chunks)
                ):
                    pbar.update(chunk_sizes[gpu_id])

                    # 主进程负责合并结果
        with FileLock(lock_file, timeout=300):
            self._merge_temp_files()
            print("所有GPU已完成，临时文件已合并")

        shutil.rmtree(temp_dir, ignore_errors=True)
        total_time = time.time() - start_time

        print(f"\n{'=' * 50}")
        print(f"处理完成: 共耗时 {total_time:.2f} 秒")
        print(f"平均速度: {len(filtered_files) / total_time:.2f} 文件/秒")

        return []
    def run(self, output_file, start_idx=None, end_idx=None):
        self.output_file = f"/mnt/h_h_public/lh/lz/Rare_rag/data/processed/text/{output_file}"
        if start_idx is not None and end_idx is not None:
            self.start_idx = start_idx
            self.end_idx = end_idx
            print(f"只处理文件索引从 {start_idx} 到 {end_idx} 的文件")

        total_start_time = time.time()
        start_time = time.time()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config['model']['embed_model'],
            encode_kwargs={'batch_size': self.config["model"]["batch_size"]}
        )
        embed_init_time = time.time() - start_time
        print(f"\n[耗时] 嵌入模型初始化: {embed_init_time:.2f}秒")
        start_time = time.time()
        self.get_vectorstore(self.embeddings, f"/mnt/h_h_public/lh/lz/Rare_rag/data/processed/text/{output_file}")
        vectorstore_time = time.time() - start_time
        print(f"[耗时] 向量文件保存: {vectorstore_time:.2f}秒")

        total_time = time.time() - total_start_time
        print(f"\n{'=' * 50}")
        print(f"[总耗时] 整个Embedding Pipeline: {total_time:.2f}秒")
if __name__ == "__main__":
    mp.set_start_method('spawn')
    print("Start！！！！！")

    # 解析命令行参数
    config_path = sys.argv[1] if len(sys.argv) > 1 else "../configs/text_rag_demo.yaml"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "embeddings.jsonl"
    start_idx = int(sys.argv[3]) if len(sys.argv) > 3 else None
    end_idx = int(sys.argv[4]) if len(sys.argv) > 4 else None
    pipeline = MyScaleRAGPipeline(config_path)
    pipeline.run(output_file=output_file, start_idx=start_idx, end_idx=end_idx)