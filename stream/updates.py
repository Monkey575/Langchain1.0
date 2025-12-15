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
    temperature=1,
    max_tokens=2000,
    timeout=None,
)

def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny in {city}!"

agent = create_agent(
    model=model,
    tools=[get_weather],
)
for chunk in agent.stream(      #chunk是一个字典，key是step，value是对应step的内容，随时收到代理的最新状态
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="updates",  #设置stream模式为updates
):
    for step, data in chunk.items():
        print(f"step: {step}")
        print(f"content: {data['messages'][-1].content_blocks}")