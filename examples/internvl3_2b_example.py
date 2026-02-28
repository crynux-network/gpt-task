import logging
import base64
from pathlib import Path

import dotenv

from gpt_task.inference import run_task

dotenv.load_dotenv()

logging.basicConfig(
    format="[{asctime}] [{levelname:<8}] {name}: {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    level=logging.INFO,
)

image_path = Path(__file__).resolve().parent / "test.png"
image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image",
                "base64": image_base64,
            },
        ],
    }
]

res = run_task(
    model="OpenGVLab/InternVL3-2B",
    messages=messages,
    generation_config={"max_new_tokens": 256},
    seed=42,
    dtype="float16",
)
print(res)
