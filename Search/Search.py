# -*- coding：utf-8 -*-
##构建一个语义搜索引擎，这是RAG的重要组成部分
#1. 加载文件
#2. 分割文件
#3. 嵌入，将文件转化为向量
#4. 建立知识库，利用被分割后的文件向量建立向量库
#5. 搜索，将用户查询转为向量后在知识库中比对，返回相似度最高的文本
#6. 封装，封装为runable接口的chain对象

import sys
sys.stdout.reconfigure(encoding='utf-8')
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import getpass
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

#加载文件,document是langchain中的一个数据容器，用于处理文件相关
file_path = "nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
#print(len(docs))

#分割文件,单一文件可能过长，不利于提取关键信息
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=20, chunk_overlap=0, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print(len(all_splits))

#嵌入，嵌入即将自然语言转化为向量，计算机无法识别自然语言，但可以识别向量
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key = api_key,
    base_url = base_url
)
vector_1 = embeddings.embed_query(all_splits[0].page_content)
#print(f"Generated vectors of length {len(vector_1)}\n")
#print(vector_1[:10])

#建立向量数据库，对与被转化为向量的Document，我们将其在向量数据库VectorStore中进行逐一比较，找出与其最匹配的文本块
vectorStore = InMemoryVectorStore(embeddings)   #实例化向量数据库，该向量数据库即将成为本次的知识库
ids = vectorStore.add_documents(documents=all_splits)   #将被分割的文件存储到向量数据库中，其中all_splits会自动被转化为向量再存储，ids是每个文本块的唯一标识符，VecotrStore在这里只是一个存储向量的容器

#搜索,（依据两个向量的相似度进行评分，再按照评分排序，其中有专门的评分函数）
#results = vectorStore.similarity_search(    #将下面文本在向量数据库中进行相似度搜索，文本会自动被转化为向量，再被搜索
#    "学校"
#)
#print("-----最相似文本块-----")
#print(results[0])

#封装为chain， vectorstore只是一个存储向量的容器，本身不支持runable接口，无法用LCEL连接，LangChain Retrievers采用vectorstore作为存储容器，实现了runable接口，这里我们手动实现一个建议的retriever
@chain
def retriever(query:str) -> (List[Document]):
    return vectorStore.similarity_search_with_score(query, k=1)

# 获取批量查询结果
results = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ]
)

# 打印每个查询的结果和得分
for query, query_results in zip(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
    results
):
    print(f"Results for query: {query}")
    for doc, score in query_results:  # `query_results` 是一个 (Document, score) 元组的列表
        print(f"Document: {doc.page_content[:100]}...")  # 只打印文档的前100个字符
        print(f"Score: {score}")
    print("-" * 80)  # 分隔线，方便查看多个查询的结果