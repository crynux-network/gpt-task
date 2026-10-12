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

res = run_task(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    generation_config={"max_new_tokens": 64},
    seed=42,
    dtype="bfloat16",
    quantization_config={
        "quant_method": "bitsandbytes",
        "quant_kwargs": {
            "load_in_8bit": True,
            "llm_int8_threshold": 6.0,
        },
    },
)
print(res)
print("execution_dtype=", get_execution_dtype())
print("execution_quantization=", get_execution_quantization())
