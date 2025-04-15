**配置文件：**

```yaml
# 路径配置
paths:
  # JSONL文件存放目录（支持多个JsonL文件）
  jsonl_dir: "../data/raw/text/"
  # 向量数据库持久化存储目录
  persist_dir: "./chroma_db"

# 模型配置
model:
  # 如果嵌入模型本地路径（需提前从https://hf-mirror.com/BAAI/bge-large-en-v1.5下载）
  embed_model: "bge-large-en-v1.5"
  # 聊天模型名称
  chat_model: "gpt-3.5-turbo"

# 文本分割器配置
text_splitter:
  # 每个文本块的最大长度
  chunk_size: 512
  # 相邻文本块的重叠长度
  chunk_overlap: 128

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
  # 身份验证密钥
  api_key: "sk-ffSAwSlifFVkCujrLcUDfamzpVgjliaFJ4zA4sEv3hT6DNe9"
  # 系统提示词
  system_prompt: "You are a helpful assistant that answers questions based on the provided context."

# 测试查询列表
queries:
  - "Human Genome Project"  # 人类基因组计划
  - "Jupiter's Water"       # 木星水资源
```

**使用案例：**

```
 python text_rag_pipeline.py 
```



```
(rare) liuzhou@ps:~/projects/Rare_rag/pipelines$ python text_rag_pipeline.py 
创建新向量数据库...
正在加载文件: ../data/raw/text/test_text.jsonl
共加载 914 篇文档

知识库统计:
- 原始文档数: 914
- 处理后片段数: 16373
- 平均片段长度: 399字符

==================================================
原始问题: Human Genome Project

检索到的相关内容:
[结果 2] (真实相似度: 0.288842)
Human Genome Project: First scientific milestone of the twenty-first century
Chris Talbot
[结果 3] (真实相似度: 0.426272)
The mapping of the human genome is a fundamental milestone in the development of science. The “letters” of this genetic code—3.1 billion DNA base pairs, equivalent to 200 telephone directories each of
[结果 4] (真实相似度: 0.462442)
Biotechnology Development for the Pediatric Cancer Genome ProjectAdvances in Next-Generation SequencingDevelopment of Novel Analytical MethodsExpanding St. Jude's Computational InfrastructureValidatin
[结果 5] (真实相似度: 0.532997)
Each genomics project will run for two years and will include sequencing of the samples and analysis of the vast amounts of data that this will produce. All the data from the projects will be placed i
[结果 6] (真实相似度: 0.537137)
Genomics and Personalized Medicine Breakthroughs

AI回答:
The Human Genome Project is a significant scientific milestone of the twenty-first century that involved mapping the human genetic code, which consists of 3.1 billion DNA base pairs. This project, led by scientists worldwide, aims to understand human genetics better and potentially revolutionize the field of medicine by paving the way for personalized treatments based on an individual's genetic makeup. The project is expected to be completed within the next three years.


==================================================
原始问题: Jupiter's Water

检索到的相关内容:
[结果 2] (真实相似度: 0.471556)
Jupiter Still Has Water from 1994 Comet Crash
By Megan Gannon
Distribution of water in the stratosphere of Jupiter as measured with ESA's Herschel space observatory …
[结果 3] (真实相似度: 0.561429)
that the comet collision not only gave Jupiter scars big enough to be seen from small telescopes on Earth; the ice-filled comet also dropped loads of water onto the atmosphere of our solar system's bi
[结果 4] (真实相似度: 0.564171)
Using ESA's Herschel space observatory, the most powerful infrared telescope ever sent into space, scientists recently mapped out the distribution of water vapor in Jupiter's upper atmosphere. They fo
[结果 5] (真实相似度: 0.573359)
Infrared Space Observatory in 1997. The spacecraft also found similar traces of water in the atmospheres of Saturn, Neptune and Uranus. On Jupiter, scientists knew that it could not have floated up fr
[结果 6] (真实相似度: 0.590966)
The trailing hemisphere appears splattered and streaked with a red material - which has caused debate for 15 years. The prevailing theory is that one of Jupiter's largest moons, Io, spews volcanic sul

AI回答:
Jupiter has water in its atmosphere, which was delivered by the collision of the comet Shoemaker-Levy 9 in 1994. This event not only left visible scars on Jupiter but also transferred loads of water onto the planet's atmosphere. Scientists discovered water vapor in Jupiter's upper atmosphere in 1997 using the European Space Agency's Infrared Space Observatory, and later mapped out the distribution of water vapor in Jupiter's upper atmosphere with ESA's Herschel space observatory. The research revealed that the southern hemisphere of Jupiter, where the comet struck, contains two to three times more water than the northern hemisphere, with most of the water clustered around the impact sites of Shoemaker-Levy 9. The asymmetry between the two hemispheres indicates that water was delivered during a specific event. Jupiter's water content could not have risen from the inner atmosphere due to a vapor-blocking "cold trap." Additionally, there are indications of other substances, such as red material and volcanic sulfur, present on Jupiter's surface.

```

