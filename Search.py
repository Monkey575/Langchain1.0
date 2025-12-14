# -*- coding：utf-8 -*-
##构建一个语义搜索引擎，这是RAG的重要组成部分
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
file_path = "123d.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
#print(len(docs))

#分割文件,单一文件可能过长，不利于提取关键信息
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=40, chunk_overlap=0, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
#print(len(all_splits))

#嵌入，嵌入即将自然语言转化为向量，计算机无法识别自然语言，但可以识别向量
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key = api_key,
    base_url = base_url
)
vector_1 = embeddings.embed_query(all_splits[0].page_content)
#print(f"Generated vectors of length {len(vector_1)}\n")
#print(vector_1[:10])

#向量数据库，对与被转化为向量的Document，我们将其在向量数据库VectorStore中进行逐一比较，找出与其最匹配的文本块
vectorStore = InMemoryVectorStore(embeddings)   #创建向量数据库
ids = vectorStore.add_documents(documents=all_splits)   #ids是每个文本块的唯一标识符
print(len(ids))
results = vectorStore.similarity_search(    #将下面文本在向量数据库中进行相似度搜索
    "学校"
)
for i in range (0,len(results)):   
    print("-----文本块-----")
    print(results[i].page_content)

print("-----最相似文本块-----")
print(results[0])   