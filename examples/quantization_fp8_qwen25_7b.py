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
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    generation_config={"max_new_tokens": 64},
    seed=42,
    dtype="bfloat16",
    quantization_config={
        "quant_method": "fp8",
        "quant_kwargs": {
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "scale_fmt": "float",
            "dequantize": False,
        },
    },
)
print(res)
print("execution_dtype=", get_execution_dtype())
print("execution_quantization=", get_execution_quantization())
