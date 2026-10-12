import logging

import dotenv
from gpt_task.inference import get_execution_dtype, get_execution_quantization, run_task

dotenv.load_dotenv()

logging.basicConfig(
    format="[{asctime}] [{levelname:<8}] {name}: {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    level=logging.INFO,
)

# Linux CUDA only. The checkpoint owns the GPTQ configuration.
res = run_task(
    model="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    generation_config={"max_new_tokens": 64},
    seed=42,
    dtype="auto",
)
print(res)
print("execution_dtype=", get_execution_dtype())
print("execution_quantization=", get_execution_quantization())
