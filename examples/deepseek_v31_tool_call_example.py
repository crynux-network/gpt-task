import json
import logging
import re
from typing import Any, Dict, List, Optional

import dotenv

from gpt_task.cache.memory_impl import MemoryModelCache
from gpt_task.inference import run_task

dotenv.load_dotenv()


logging.basicConfig(
    format="[{asctime}] [{levelname:<8}] {name}: {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    level=logging.DEBUG,
)


def get_current_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
    return {
        "location": location,
        "temperature": "24",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }


def calculator(operation: str, x: float, y: float) -> Dict[str, Any]:
    operations = {
        "add": lambda: x + y,
        "subtract": lambda: x - y,
        "multiply": lambda: x * y,
        "divide": lambda: x / y if y != 0 else "Error: Division by zero",
    }
    return {
        "operation": operation,
        "result": operations.get(operation, lambda: "Invalid operation")(),
    }


def handle_tool_call(tool_call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        function_name = tool_call.get("function", {}).get("name")
        arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))

        if function_name == "get_current_weather":
            return get_current_weather(**arguments)
        if function_name == "calculator":
            return calculator(**arguments)
        return None
    except Exception as exc:
        logging.error(f"Error handling tool call: {exc}")
        return None


def extract_tool_calls(response_content: str) -> List[Dict[str, Any]]:
    tool_calls: List[Dict[str, Any]] = []

    # Format 1: <tool_call>{"name":"...","arguments":{...}}</tool_call>
    classic_pattern = r"<tool_call>\s*({[\s\S]*?})\s*</tool_call>"
    for match in re.findall(classic_pattern, response_content):
        try:
            payload = json.loads(re.sub(r"\s+", " ", match).strip())
            tool_calls.append(
                {
                    "function": {
                        "name": payload.get("name", ""),
                        "arguments": json.dumps(payload.get("arguments", {})),
                    },
                    "id": f"call_{len(tool_calls)}",
                }
            )
        except json.JSONDecodeError:
            continue

    # Format 2: DeepSeek tool tags:
    # <｜tool▁call▁begin｜>name<｜tool▁sep｜>{...}<｜tool▁call▁end｜>
    deepseek_pattern = (
        r"<\uFF5Ctool\u2581call\u2581begin\uFF5C>"
        r"\s*(.*?)\s*"
        r"<\uFF5Ctool\u2581sep\uFF5C>"
        r"\s*({[\s\S]*?})\s*"
        r"<\uFF5Ctool\u2581call\u2581end\uFF5C>"
    )
    for function_name, arguments in re.findall(deepseek_pattern, response_content):
        try:
            parsed_args = json.loads(arguments)
            tool_calls.append(
                {
                    "function": {
                        "name": function_name.strip(),
                        "arguments": json.dumps(parsed_args),
                    },
                    "id": f"call_{len(tool_calls)}",
                }
            )
        except json.JSONDecodeError:
            continue

    return tool_calls


def process_conversation(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    history = messages.copy()
    model_cache = MemoryModelCache()

    while True:
        response = run_task(
            model="deepseek-ai/DeepSeek-V3.1",
            messages=history,
            tools=tools,
            generation_config={
                "repetition_penalty": 1.1,
                "do_sample": True,
                "temperature": 0.7,
                "max_new_tokens": 32768,
            },
            # DeepSeek V3.1 tool calling is in non-thinking mode.
            template_args={"thinking": False},
            seed=42424422,
            dtype="float16",
            model_cache=model_cache,
        )

        if not response or "choices" not in response or not response["choices"]:
            logging.error("No choices found in response")
            break

        choice = response["choices"][0]
        if not isinstance(choice, dict) or "message" not in choice or "content" not in choice["message"]:
            logging.error("Unexpected response format in choices")
            break

        assistant_content = choice["message"]["content"] or ""
        tool_calls = extract_tool_calls(assistant_content)

        if not tool_calls:
            history.append({"role": "assistant", "content": assistant_content})
            break

        clean_content = re.sub(r"<tool_call>\s*(?:{[\s\S]*?})\s*</tool_call>", "", assistant_content).strip()
        if clean_content:
            history.append({"role": "assistant", "content": clean_content})

        for tool_call in tool_calls:
            tool_result = handle_tool_call(tool_call)
            history.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": json.dumps(tool_result) if tool_result else "Error executing function",
                }
            )

    return history


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, for example San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform basic mathematical operations",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The mathematical operation to perform",
                    },
                    "x": {"type": "number", "description": "The first number"},
                    "y": {"type": "number", "description": "The second number"},
                },
                "required": ["operation", "x", "y"],
            },
        },
    },
]

messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant that can perform basic mathematical operations and get "
            "the current weather in a given location using the given tools."
        ),
    },
    {"role": "user", "content": "What is the weather like in Tokyo and what is 2+2?"},
]

conversation_history = process_conversation(messages, tools)

print("Final conversation:\n")
for message in conversation_history:
    if message["role"] == "user":
        print(f"\nUser: {message['content']}")
    elif message["role"] == "assistant":
        if message.get("tool_calls"):
            print(f"Assistant: (Calling function: {message['tool_calls'][0]['function']['name']})")
        else:
            print(f"Assistant: {message['content']}")
    elif message["role"] == "tool":
        print(f"Tool response: {message['content']}")
    elif message["role"] == "system":
        print(f"System: {message['content']}")
