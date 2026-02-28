import logging
import time

import dotenv

from gpt_task.cache import MemoryModelCache
from gpt_task.inference import run_task

dotenv.load_dotenv()


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

total_start = time.perf_counter()

for index, messages in enumerate(all_messages, start=1):
    step_start = time.perf_counter()

    res = run_task(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=messages,
        generation_config={"max_new_tokens": 32768},
        seed=42,
        model_cache=cache,
        dtype="float16"
    )

    step_elapsed = time.perf_counter() - step_start
    total_elapsed = time.perf_counter() - total_start

    usage = res.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    finish_reason = None
    if res.get("choices"):
        finish_reason = res["choices"][0].get("finish_reason")

    tokens_per_sec = completion_tokens / step_elapsed if step_elapsed > 0 else 0.0

    print(f"\nStep {index}")
    print(f"  prompt={messages[0]['content']}")
    print(f"  step_elapsed_sec={step_elapsed:.3f}")
    print(f"  total_elapsed_sec={total_elapsed:.3f}")
    print(f"  completion_tokens={completion_tokens}")
    print(f"  total_tokens={total_tokens}")
    print(f"  finish_reason={finish_reason}")
    print(f"  tokens_per_sec={tokens_per_sec:.3f}")
