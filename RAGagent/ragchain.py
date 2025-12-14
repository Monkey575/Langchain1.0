#构建一个RAGChain
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
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent

load_dotenv()
base_url = os.getenv("BASE_URL")
api_key = os.getenv("API_KEY")

#实例化聊天模型
model = init_chat_model(
    model="gpt-4o-mini",
    api_key=api_key,
    base_url=base_url
)
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
##21. RAGagent：实现一个简单的RAGChain
###211. 实现一个动态提示词中间件，在每次用户查询时，对该查询检索知识库，并将检索到的内容注入到提示词中
@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text #获取用户的最后一条查询
    retrieved_docs = vector_store.similarity_search(last_query) #检索

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)  #将检索内容序列化

    system_message = (  #构建系统消息
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )

    return system_message
agent = create_agent(model, tools=[], middleware=[prompt_with_context])
####212. 使用RAGChain进行查询
query = "What is task decomposition?"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()