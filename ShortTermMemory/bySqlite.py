from langchain.chat_models import init_chat_model
from langchain.agents import create_agent, AgentState
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver  # 1. 导入 SQLite 检查点
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

# 初始化模型
model = init_chat_model(
    model="gpt-4o-mini",
    api_key=api_key,
    base_url=base_url,
    temperature=1,
)

# 定义自定义的 AgentState，可以独立标识某些特殊信息，方便agent直接读取
class myAgentState(AgentState):
    userid:str
    perference:str

# 定义一个简单的工具（为了代码能运行，模拟你的 get_user_info）
def get_weather(city: str) -> str:
    """查询天气的工具"""
    return "晴天"

# 2. 指定 SQLite 数据库文件路径 (会自动在当前目录下创建 memory.sqlite)
DB_PATH = "memory.sqlite"

# 3. 使用上下文管理器打开 SQLite 连接
# 注意：所有的 Agent 操作必须在 with 缩进块内部进行，否则数据库连接会关闭
with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
    
    # 创建 Agent，传入 checkpointer
    agent = create_agent(
        model=model,
        tools=[get_weather],
        checkpointer=checkpointer,
        state_schema=myAgentState,  
    )

    # 配置线程 ID（同一个 thread_id 会读取同一份历史记录）
    config = {"configurable": {"thread_id": "1"}}

    print("--- 第一次对话 (流式只看内容) ---")
    
    # 第一次对话
    # 注意：这里使用 token, _ 来解包元组，避免报错
    for token, _ in agent.stream(
        {"messages": [{"role": "user", "content": "你好，我是Bob，请记住我的名字,我的id是1，我喜欢晴天。"}]},
        config,
        stream_mode="messages",
    ):
        if token.content:
            print(token.content, end="", flush=True)
    
    print("\n\n--- 第二次对话 (测试记忆) ---")

    # 第二次对话
    for token, _ in agent.stream(
        {"messages": [{"role": "user", "content": "我叫什么名字？，喜欢什么天气？"}]},
        config,   
        stream_mode="messages",
    ):
        if token.content:
            print(token.content, end="", flush=True)

    print("\n")