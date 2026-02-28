import logging

import dotenv

from gpt_task.inference import run_task

dotenv.load_dotenv()


logging.basicConfig(
    format="[{asctime}] [{levelname:<8}] {name}: {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    level=logging.INFO,
)

messages = [{"role": "user", "content": "I want to create a chat bot. Any suggestions?"}]


res = run_task(
    model="deepseek-ai/DeepSeek-V3.2",
    messages=messages,
    generation_config={"max_new_tokens": 32768},
    template_args={"thinking_mode": "thinking"},
    seed=42,
    dtype="float16",
)
print(res)
