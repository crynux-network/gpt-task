import logging
import json
import re
import sys
from typing import List, Dict, Any, Optional

from gpt_task.inference import run_task
from gpt_task.cache.memory_impl import MemoryModelCache
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

def get_current_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
    """Get the current weather in a given location"""
    # Mock weather data
    return {
        "location": location,
        "temperature": "24",
        "unit": unit,
        "forecast": ["sunny", "windy"]
    }

def calculator(operation: str, x: float, y: float) -> Dict[str, Any]:
    """Perform a mathematical operation on two numbers"""
    operations = {
        "add": lambda: x + y,
        "subtract": lambda: x - y,
        "multiply": lambda: x * y,
        "divide": lambda: x / y if y != 0 else "Error: Division by zero"
    }
    return {
        "operation": operation,
        "result": operations.get(operation, lambda: "Invalid operation")()
    }

def handle_tool_call(tool_call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handle tool calls and return the result"""
    try:
        function_name = tool_call.get("function", {}).get("name")
        arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))

        if function_name == "get_current_weather":
            return get_current_weather(**arguments)
        elif function_name == "calculator":
            return calculator(**arguments)
        else:
            return None
    except Exception as e:
        logging.error(f"Error handling tool call: {e}")
        return None

def extract_tool_calls(response_content: str) -> List[Dict[str, Any]]:
    """Extract tool calls from the response content using regex pattern matching."""
    tool_calls = []
    pattern = r'<tool_call>\s*({[\s\S]*?})\s*</tool_call>'
    matches = re.findall(pattern, response_content)

    for match in matches:
        try:
            clean_json = re.sub(r'\s+', ' ', match).strip()
            tool_call_data = json.loads(clean_json)
            formatted_tool_call = {
                "function": {
                    "name": tool_call_data.get("name", ""),
                    "arguments": json.dumps(tool_call_data.get("arguments", {}))
                },
                "id": f"call_{len(tool_calls)}"
            }
            tool_calls.append(formatted_tool_call)
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing tool call JSON: {e}")
            logging.error(f"Problematic JSON string: {match}")

    return tool_calls

def process_conversation_streaming(messages: List[Dict[str, str]], tools: List[Dict[str, Any]]):
    """Process the conversation with tool calling support and streaming output"""
    conversation_history = messages.copy()
    model_cache = MemoryModelCache()

    # Initialize total usage at the beginning of your function
    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }

    current_session_usage = None
    accumulated_content = ""

    def stream_callback(chunk: GPTTaskStreamResponse):
        nonlocal accumulated_content
        nonlocal current_session_usage

        if 'choices' in chunk and len(chunk['choices']) > 0:
            delta = chunk['choices'][0].get('delta', {})

            # Handle different types of delta content
            if 'content' in delta:
                content = delta['content']
                if content:
                    sys.stdout.write(content)
                    sys.stdout.flush()
                    accumulated_content += content

            finish_reason = chunk['choices'][0].get('finish_reason')
            if finish_reason:
                sys.stdout.write('\n')
                sys.stdout.flush()

            # Update usage information if present in the chunk
            if chunk.get('usage'):
                current_session_usage = chunk['usage']

    try:
        run_task(
            model="Qwen/Qwen3-8B",
            messages=conversation_history,
            tools=tools,
            stream_callback=stream_callback,
            generation_config={
                "repetition_penalty": 1.1,
                "do_sample": True,
                "temperature": 0.7,
                "max_new_tokens": 32768,
            },
            seed=42424422,
            dtype="float16",
            model_cache=model_cache
        )

        total_usage["prompt_tokens"] += current_session_usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += current_session_usage.get("completion_tokens", 0)
        total_usage["total_tokens"] += current_session_usage.get("total_tokens", 0)

         # Add the assistant's message without tool calls
        clean_content = re.sub(r'<tool_call>\s*(?:{[\s\S]*?})\s*</tool_call>', '', accumulated_content).strip()
        if clean_content:
            conversation_history.append({"role": "assistant", "content": clean_content})

        # After streaming is complete, process any tool calls
        tool_calls = extract_tool_calls(accumulated_content)

        if tool_calls:
            print("\nExecuting tool calls...")

            # Process each tool call
            for tool_call in tool_calls:
                print(f"\nCalling function: {tool_call['function']['name']}")
                tool_result = handle_tool_call(tool_call)

                # Add the tool call to conversation history
                conversation_history.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call]
                })

                # Process and display tool response
                result_str = json.dumps(tool_result) if tool_result else "Error executing function"
                print(f"Tool response: {result_str}")

                # Add tool response to conversation history
                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": result_str
                })

            print("\nGenerating final response with tool results...")

            accumulated_content = ""
            current_session_usage = None

            run_task(
                model="Qwen/Qwen3-8B",
                messages=conversation_history,
                stream_callback=stream_callback,
                generation_config={
                    "repetition_penalty": 1.1,
                    "do_sample": True,
                    "temperature": 0.7,
                    "max_new_tokens": 32768,
                },
                seed=42424422,
                dtype="float16",
                model_cache=model_cache
            )

            total_usage["prompt_tokens"] += current_session_usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += current_session_usage.get("completion_tokens", 0)
            total_usage["total_tokens"] += current_session_usage.get("total_tokens", 0)

            # Add final response to conversation history
            conversation_history.append({
                "role": "assistant",
                "content": accumulated_content
            })

        return conversation_history, total_usage

    except Exception as e:
        _logger.exception("Error during streaming generation")
        raise

# Define the function schemas
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
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use"
                    }
                },
                "required": ["location"]
            }
        }
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
                        "description": "The mathematical operation to perform"
                    },
                    "x": {
                        "type": "number",
                        "description": "The first number"
                    },
                    "y": {
                        "type": "number",
                        "description": "The second number"
                    }
                },
                "required": ["operation", "x", "y"]
            }
        }
    }
]

if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can perform basic mathematical operations and get the current weather in a given location using the given tools."},
        {"role": "user", "content": "What's the weather like in Tokyo? and what is 2+2? You must use the tools to answer the question."}
    ]

    print("Starting streaming conversation...\n")
    conversation_history, usage = process_conversation_streaming(messages, tools)

    print("\nFinal conversation history:")
    for message in conversation_history:
        if message["role"] == "user":
            print(f"\nUser: {message['content']}")
        elif message["role"] == "assistant":
            if message.get("tool_calls"):
                for tool_call in message['tool_calls']:
                    print(f"\nAssistant: Calling {tool_call['function']['name']} with arguments: {tool_call['function']['arguments']}")
            elif message.get("content"):
                print(f"\nAssistant: {message['content']}")
        elif message["role"] == "tool":
            print(f"Tool response: {message['content']}")
        elif message["role"] == "system":
            print(f"System: {message['content']}")

    print("\nToken Usage:")
    if usage:
        print(f"  Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
        print(f"  Completion tokens: {usage.get('completion_tokens', 'N/A')}")
        print(f"  Total tokens: {usage.get('total_tokens', 'N/A')}")
    else:
        print("  No usage data available")
