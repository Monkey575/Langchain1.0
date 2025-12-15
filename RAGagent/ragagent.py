#构建一个RAG智能体
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import getpass
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()
base_url = os.getenv("BASE_URL")
api_key = os.getenv("API_KEY")

#实例化聊天模型
model = init_chat_model(
    model="gpt-4o-mini",
    api_key=api_key,
    base_url=base_url
)
DB_PATH = "memory.sqlite"
config = {"configurable": {"thread_id": "2"}}  #配置线程 ID（同一个 thread_id 会读取同一份历史记录）
#连接数据库
with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
    #实例化嵌入模型
    embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-small",
        api_key = api_key,
        base_url = base_url
    )
    #选择向量存储器
    vector_store = InMemoryVectorStore(embeddings)
    #1. Index 索引
    ##11. 加载文档
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))   #爬取网页
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    assert len(docs) == 1
    ##12. 拆分文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)
    ##13. 文档嵌入与向量存储
    document_ids = vector_store.add_documents(documents=all_splits) #将被分割的文件存储到向量数据库中，建立知识库

    #2. Retrieval and Generation 检索与生成
    ##21. RAGagent：实现一个简单的RAG智能体
    ###211. 实现一个tool，用于从知识库中检索相关信息
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""
        retrieved_docs = vector_store.similarity_search(query, k=2) #检索
        serialized = "\n\n".join(                                   #序列化检索结果
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")#metadata是文档的背景，page_content是文档的内容,在RAG构建中，metadata返回的文档的来源，以便在生成答案时引用这些信息
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    ###212. 创建RAG智能体
    tools = [retrieve_context]    #将tool添加到工具列表中
    prompt = (                    #构建提示词
        "You have access to a tool that retrieves context from a blog post. "
        "Use the tool to help answer user queries."
        )
    agent = create_agent(         #构建agent，将大模型，工具，提示词结合在一起
        model=model,
        tools=tools,
        system_prompt=prompt,
        checkpointer=checkpointer
    )
    query = (                     #用户查询
        "任务分解的标准方法是什么?\n\n"
        "得到答案后，查找该方法的常见扩展."
    )
    print("\n--- 问题1 ---\n")
    for token, _ in agent.stream(    #查看输出流
        {"messages": [{"role": "user", "content": query}]},
        config,
        stream_mode="messages",
    ):
        if token.content:
            print(token.content, end="", flush=True)

    print("\n\n--- 问题2 ---\n")
    for token, _ in agent.stream(    #查看输出流
        {"messages": [{"role": "user", "content": "你如何评判该方法的优劣？"}]},
        config,
        stream_mode="messages",
    ):
        if token.content:
            print(token.content, end="", flush=True)