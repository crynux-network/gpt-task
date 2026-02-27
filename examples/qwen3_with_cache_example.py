import logging
from gpt_task.inference import run_task
from gpt_task.cache import MemoryModelCache


logging.basicConfig(
    format="[{asctime}] [{levelname:<8}] {name}: {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    level=logging.INFO,
)

cache = MemoryModelCache()


all_messages = [
    [{"role": "user", "content": "I want to create a chat bot. Any suggestions?"}],
    [{"role": "user", "content": "What is the highest mountain in the world?"}],
    [{"role": "user", "content": "I have a dream."}],
    [{"role": "user", "content": "It's raining today."}],
]


for messages in all_messages:
    res = run_task(
        model="Qwen/Qwen3-8B",
        messages=messages,
        seed=42,
        model_cache=cache,
    )
    print(res)
