import logging
import sys
import time

from gpt_task.inference import run_task
from gpt_task.models import GPTTaskStreamResponse
import dotenv
dotenv.load_dotenv()

logging.basicConfig(
    format="[{asctime}] [{levelname:<8}] {name}: {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    level=logging.INFO,
)

_logger = logging.getLogger(__name__)

messages = [
    {"role": "user", "content": "Write a short story about a magical forest."}
]

print("Starting generation...", flush=True)

# Enable streaming output
usage = None

try:
    def stream_callback(chunk: GPTTaskStreamResponse):
        global usage  # Declare usage as nonlocal
        content = chunk["choices"][0]["delta"].get("content", "")
        finish_reason = chunk["choices"][0].get("finish_reason")
        chunk_usage = chunk["usage"]

        _logger.debug(f"Received chunk: {chunk}")

        if content:
            sys.stdout.write(content)
            sys.stdout.flush()

        if finish_reason:
            sys.stdout.write("\n")
            sys.stdout.flush()
            _logger.debug(f"Generation finished with reason: {finish_reason}")

        if chunk_usage:
            usage = chunk_usage

    run_task(
        model="Qwen/Qwen3-8B",
        messages=messages,
        stream_callback=stream_callback,
        generation_config={
            "repetition_penalty": 1.1,
            "do_sample": True,
            "temperature": 0.7,
            "max_new_tokens": 32768
        },
        seed=42,
        dtype="float16"
    )

except Exception as e:
    _logger.exception("Error during generation")
    raise

print(f"Usage: {usage}")

print("\nGeneration complete!", flush=True)
