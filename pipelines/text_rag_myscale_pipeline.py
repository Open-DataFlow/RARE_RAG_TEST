from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import MyScale
import os
import yaml
import http.client
import json
import glob
from typing import List, Dict, Any


class MyScaleRAGPipeline:
    def __init__(self, config_path: str = "../configs/text_rag_demo.yaml"):
        self.config = self._load_config(config_path)
        self._setup_environment_vars()

    def _setup_environment_vars(self):
        os.environ['MYSCALE_HOST'] = self.config['myscale']['host']
        os.environ['MYSCALE_PORT'] = str(self.config['myscale']['port'])
        os.environ['MYSCALE_USERNAME'] = self.config['myscale']['username']
        os.environ['MYSCALE_PASSWORD'] = self.config['myscale']['password']

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

    def get_vectorstore(self, embeddings: Any) -> MyScale:
        """初始化MyScale向量数据库"""
        # 添加元数据到文档
        docs = self.load_all_jsonl_documents()
        splits = self.process_documents(docs)

        for doc in splits:
            if not hasattr(doc, 'metadata'):
                doc.metadata = {}
            doc.metadata.update({
                "source": doc.metadata.get("source", "unknown"),
                "chunk_id": hash(doc.page_content)  # 为每个分块生成唯一ID
            })

        # 打印统计信息
        total_chunks = len(splits)
        avg_chunk_len = sum(len(d.page_content) for d in splits) / total_chunks
        print(f"\n知识库统计:")
        print(f"- 原始文档数: {len(docs)}")
        print(f"- 处理后片段数: {total_chunks}")
        print(f"- 平均片段长度: {avg_chunk_len:.0f}字符")

        # 创建MyScale向量存储
        vectorstore = MyScale.from_documents(
            documents=splits,
            embedding=embeddings,
            config={
                "table": self.config['myscale'].get('table_name', 'default_rag_table'),
                "database": self.config['myscale'].get('database', 'default'),
                "index_type": self.config['myscale'].get('index_type', 'IVFFLAT'),
                "metric_type": self.config['myscale'].get('metric_type', 'COSINE')
            }
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
            context.append(f"【文档片段 {i}】(来源: {doc.metadata.get('source', '未知')})\n{doc.page_content}\n")
        return "\n".join(context)

    def run(self):
        """运行整个RAG流程"""
        # 初始化嵌入模型
        embeddings = HuggingFaceEmbeddings(
            model_name=self.config['model']['embed_model']
        )

        # 获取MyScale向量数据库
        print("连接到MyScale向量数据库...")
        vectorstore = self.get_vectorstore(embeddings)

        # 配置检索器
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": self.config['retriever']['k'],
                # 可以添加MyScale特有的过滤条件
                # "where_str": "metadata.some_field = 'some_value'"
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

            # 2. 打印检索结果（带相似度分数）
            print("\n检索到的相关内容:")
            scored_results = vectorstore.similarity_search_with_score(query, k=self.config['retriever']['k'])
            for i, (doc, score) in enumerate(scored_results, 1):
                print(f"[结果 {i}] (相似度分数: {score:.4f})")
                print(f"来源: {doc.metadata.get('source', '未知')}")
                print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                print("-" * 50)

            # 3. 调用Chat API
            context = self.format_context(results)
            answer = self.query_chat_api(context, query)

            # 4. 打印最终回答
            print("\nAI回答:")
            print(f"{answer}\n")


if __name__ == "__main__":
    pipeline = MyScaleRAGPipeline()
    pipeline.run()