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

for chunk in model.stream("来一首唐诗"):
    print(chunk.content, end=" ", flush=True)