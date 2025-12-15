#这里采用内存存储，只能在单词进程中执行，仅用于演示短期记忆效果
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from langchain.agents.middleware import SummarizationMiddleware
import os
from langchain_core.runnables import RunnableConfig

load_dotenv()
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

# ... [初始化 model 和 agent 的代码保持不变] ...
model = init_chat_model(
    model="gpt-4o-mini",
    api_key=api_key,
    base_url=base_url,
    temperature=1,
    max_tokens=2000,
    timeout=None,
)

# 注意：LangChain/LangGraph 的最新版本可能更推荐使用 AgentExecutor
agent = create_agent(
    model=model,
    checkpointer=InMemorySaver(),  # 指定检查点，实现短期记忆
    middleware=[
        SummarizationMiddleware(    #摘要中间件
            model=model,
            trigger=("tokens", 4000),
            keep=("messages", 20)
        )
    ],
)
config: RunnableConfig = {"configurable": {"thread_id": "1"}}   #配置线程 ID,采用runnable接口
print("--- 第一次对话流式输出 ---")
# 第一次对话
# 修改这里：使用 token, metadata 或者 token, _ 进行解包
for token, metadata in agent.stream(  
    {"messages": [{"role": "user", "content": "my name is Bob"}]},
    config,
    stream_mode="messages",
):
    # 注意：标准 LangChain 消息块通常用 .content，但在你的环境中如果是 .content_blocks 可保持原样
    # 为了通用性，通常建议打印 token.content
    print(f"content: {token.content}") 
    # print(f"content: {token.content_blocks}") # 如果你确定你的模型对象有这个属性保留这行
    print("\n")
    

print("\n\n--- 第二次对话流式输出(只看内容) ---")
# 第二次对话
for token, _ in agent.stream(  
    {"messages": [{"role": "user", "content": "What is my name?"}]},
    config,
    stream_mode="messages",
):
    if token.content:
        print(token.content, end="", flush=True)

print("\n") # 最后换个行，保持美观

print("\n\n--- 第二次对话流式输出(只看内容) ---")
# 第二次对话
for token, _ in agent.stream(  
    {"messages": [{"role": "user", "content": "tell me what i have sayed before?"}]},
    config,
    stream_mode="messages",
):
    if token.content:
        print(token.content, end="", flush=True)