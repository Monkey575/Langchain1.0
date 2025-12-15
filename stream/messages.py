from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

model = init_chat_model(
    model="gpt-4o-mini",
    api_key=api_key,
    base_url=base_url,
)

def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny in {city}!"

agent = create_agent(
    model=model,
    tools=[get_weather],
)
for token, metadata in agent.stream(  #token是一个文本片段，这就是为什么用户可以看到ai的消息一个字一个字的输出， metadata是元数据，表明背景信息
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="messages",
):
    print(f"node: {metadata['langgraph_node']}")
    print(f"content: {token.content_blocks}")
    print("\n")