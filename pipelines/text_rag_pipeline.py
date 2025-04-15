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


class RAGPipeline:
    def __init__(self, config_path: str = "../configs/text_rag_demo.yaml"):
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_jsonl_documents(self, file_path: str) -> List[Any]:
        """加载JSONL数据"""
        loader = JSONLoader(
            file_path=file_path,
            jq_schema='.text',
            text_content=False,
            json_lines=True
        )
        return loader.load()

    def process_documents(self, docs: List[Any]) -> List[Any]:
        """处理文档分割"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['text_splitter']['chunk_size'],
            chunk_overlap=self.config['text_splitter']['chunk_overlap'],
            add_start_index=True
        )
        return text_splitter.split_documents(docs)

    def load_all_jsonl_documents(self) -> List[Any]:
        """加载目录下所有JSONL文件"""
        docs = []
        jsonl_files = glob.glob(os.path.join(self.config['paths']['jsonl_dir'], "*.jsonl"))

        if not jsonl_files:
            raise ValueError(f"目录 {self.config['paths']['jsonl_dir']} 中没有找到JSONL文件")

        for file_path in jsonl_files:
            print(f"正在加载文件: {file_path}")
            docs.extend(self.load_jsonl_documents(file_path))

        print(f"共加载 {len(docs)} 篇文档")
        return docs

    def get_vectorstore(self, embeddings: Any) -> Chroma:
        """初始化带持久化的向量数据库"""
        if os.path.exists(self.config['paths']['persist_dir']):
            print("检测到已有向量数据库，直接加载...")
            return Chroma(
                persist_directory=self.config['paths']['persist_dir'],
                embedding_function=embeddings
            )
        else:
            print("创建新向量数据库...")
            # 修改为加载整个目录
            docs = self.load_all_jsonl_documents()
            splits = self.process_documents(docs)

            # 打印统计信息
            total_chunks = len(splits)
            avg_chunk_len = sum(len(d.page_content) for d in splits) / total_chunks
            print(f"\n知识库统计:")
            print(f"- 原始文档数: {len(docs)}")
            print(f"- 处理后片段数: {total_chunks}")
            print(f"- 平均片段长度: {avg_chunk_len:.0f}字符")

            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=self.config['paths']['persist_dir']
            )
            return vectorstore

    def query_chat_api(self, context: str, question: str) -> str:
        """调用ChatAnywhere API获取回答"""
        conn = http.client.HTTPSConnection(self.config['api']['endpoint'])

        payload = json.dumps({
            "model": self.config['model']['chat_model'],
            "messages": [
                {
                    "role": "system",
                    "content": self.config['api']['system_prompt']
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ]
        })

        headers = {
            'Authorization': f"Bearer {self.config['api']['api_key']}",
            'Content-Type': 'application/json'
        }

        try:
            conn.request("POST", self.config['api']['path'], payload, headers)
            res = conn.getresponse()
            data = res.read()
            response = json.loads(data.decode("utf-8"))
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"API调用失败: {str(e)}"
        finally:
            conn.close()

    def format_context(self, docs: List[Any]) -> str:
        """格式化检索结果为上下文"""
        context = []
        for i, doc in enumerate(docs, 1):
            context.append(f"【文档片段 {i}】\n{doc.page_content}\n")
        return "\n".join(context)

    def run(self):
        """运行整个RAG流程"""
        # 初始化嵌入模型
        embeddings = HuggingFaceEmbeddings(
            model_name=self.config['model']['embed_model']
        )

        # 获取/创建向量数据库
        vectorstore = self.get_vectorstore(embeddings)

        # 配置检索器
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": self.config['retriever']['k'],
                # "score_threshold": self.config['retriever']['score_threshold']
            }
        )

        # 处理每个查询
        for query in self.config['queries']:
            print(f"\n{'=' * 50}")
            print(f"原始问题: {query}")

            # 1. 检索相关文档
            results = retriever.invoke(query)

            if not results:
                print("未找到相关结果")
                continue

            # 2. 打印检索结果
            print("\n检索到的相关内容:")
            for i, doc in enumerate(results, 1):
                _, real_score = vectorstore.similarity_search_with_score(query, k=self.config['retriever']['k'])[i-1]
                print(f"[结果 {i + 1}] (真实相似度: {real_score:.6f})")
                print(doc.page_content[:200])

            # 3. 调用Chat API
            context = self.format_context(results)
            answer = self.query_chat_api(context, query)

            # 4. 打印最终回答
            print("\nAI回答:")
            print(f"{answer}\n")


if __name__ == "__main__":
    pipeline = RAGPipeline()

    pipeline.run()