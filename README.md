## GPT Task

A unified framework to define and execute LLM and VLM generation tasks.


### Features

* Unified `run_task` entrypoint for both text-only and multimodal models
* Block-based chat messages with text and image content
* Base64-only image input for multimodal requests
* Streaming callback support with stable ChatGPT-style response shape
* Model quantizing (INT4 or INT8)
* Fine-grained generation argument control
* **RTX 50 series Support** - supports NVIDIA RTX 50 series graphics cards


### Example (LLM)

Here is an example of Qwen3-8B text generation:

```python
import logging
from gpt_task.inference import run_task


logging.basicConfig(
    format="[{asctime}] [{levelname:<8}] {name}: {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    level=logging.INFO,
)

messages = [{"role": "user", "content": "I want to create a chat bot. Any suggestions?"}]


res = run_task(
    model="Qwen/Qwen3-8B",
    messages=messages,
    seed=42,
)
print(res)
```

### Example (VLM)

```python
import base64
from pathlib import Path
from gpt_task.inference import run_task

image_path = Path("./examples/test.png")
image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image", "base64": image_base64},
        ],
    }
]

res = run_task(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    messages=messages,
    seed=42,
    dtype="float16",
)
print(res)
```


### Get started

Create and activate the virtual environment:
```shell
$ python -m venv ./venv
$ source ./venv/bin/activate
```

Install the dependencies and the library:
```shell
(venv) $ pip install -r requirements_cuda.txt && pip install -e .
```

Check and run the examples:
```shell
(venv) $ python ./examples/qwen2.5_example.py
(venv) $ python ./examples/qwen2_5_vl_3b_example.py
(venv) $ python ./examples/internvl3_2b_example.py
(venv) $ python ./examples/smolvlm_instruct_example.py
```

More explanations can be found in the doc:

[https://docs.crynux.ai/application-development/gpt-task](https://docs.crynux.ai/application-development/gpt-task)

### Task Definition

The complete task definition is `GPTTaskArgs` in the file [```./src/gpt_task/models/args.py```](src/gpt_task/models/args.py)

### Task Response

The task response definition is `GPTTaskResponse` in the file [```./src/gpt_task/models/args.py```](src/gpt_task/models/args.py)

### JSON Schema

The JSON schemas for the tasks could be used to validate the task arguments by other projects.
The schemas are given under [```./schema```](./schema). Projects could use the URL to load the JSON schema files directly.
