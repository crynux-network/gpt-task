import logging

from gpt_task.config import Config, ProxyConfig, DataDirConfig, ModelsDirConfig
from gpt_task.inference import run_task

logging.basicConfig(
    format="[{asctime}] [{levelname:<8}] {name}: {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    level=logging.INFO,
)

messages = [
    {"role": "user", "content": "I want to create a chat bot. Any suggestions?"}
]


res = run_task(
    model="Qwen/Qwen3-8B",
    messages=messages,
    seed=42,
    config=Config(
        data_dir=DataDirConfig(models=ModelsDirConfig(huggingface=".cache")),
        proxy=ProxyConfig(host="http://127.0.0.1", port=8080)
    )
)
print(res)
