import logging
import json
import re
from typing import List, Dict, Any, Optional

from gpt_task.inference import run_task
from gpt_task.cache.memory_impl import MemoryModelCache

import dotenv
dotenv.load_dotenv()

logging.basicConfig(
    format="[{asctime}] [{levelname:<8}] {name}: {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="{",
    level=logging.DEBUG,
)

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

    # Pattern to match tool calls in the format <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    # Updated to handle newlines and whitespace between tags and content
    pattern = r'<tool_call>\s*({[\s\S]*?})\s*</tool_call>'

    # Find all matches
    matches = re.findall(pattern, response_content)

    for match in matches:
        try:
            # Clean up the JSON string by removing newlines and extra whitespace
            clean_json = re.sub(r'\s+', ' ', match).strip()

            # Parse the JSON content
            tool_call_data = json.loads(clean_json)

            # Format it to match the expected structure in handle_tool_call
            formatted_tool_call = {
                "function": {
                    "name": tool_call_data.get("name", ""),
                    "arguments": json.dumps(tool_call_data.get("arguments", {}))
                },
                "id": f"call_{len(tool_calls)}"  # Generate a simple ID
            }

            tool_calls.append(formatted_tool_call)
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing tool call JSON: {e}")
            logging.error(f"Problematic JSON string: {match}")

    return tool_calls

def process_conversation(messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Process the conversation with tool calling support"""
    conversation_history = messages.copy()

    model_cache = MemoryModelCache()

    while True:
        response = run_task(
            model="Qwen/Qwen3-8B",
            messages=conversation_history,
            tools=tools,
            generation_config={
                "repetition_penalty": 1.1,
                "do_sample": True,
                "temperature": 0.7,
            },
            seed=42424422,
            dtype="float16",
            model_cache=model_cache
        )

        # Extract the assistant's message content from the choices
        if response and 'choices' in response and len(response['choices']) > 0:
            choice = response['choices'][0]
            if isinstance(choice, dict) and 'message' in choice and 'content' in choice['message']:
                assistant_content = choice['message']['content']

                # Extract tool calls from the content
                tool_calls = extract_tool_calls(assistant_content)

                # If no tool calls were found, just add the response to history
                if not tool_calls:
                    conversation_history.append({"role": "assistant", "content": assistant_content})
                    break

                # Add final assistant message without the tool calls
                # Remove tool call tags and add as regular content
                clean_content = re.sub(r'<tool_call>\s*(?:{[\s\S]*?})\s*</tool_call>', '', assistant_content).strip()
                if clean_content:
                    conversation_history.append({"role": "assistant", "content": clean_content})

                # Handle each extracted tool call
                for tool_call in tool_calls:
                    tool_result = handle_tool_call(tool_call)

                    # Add the function call to conversation history
                    conversation_history.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call]
                    })

                    # Add the function response to conversation history
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "content": json.dumps(tool_result) if tool_result else "Error executing function"
                    })

            else:
                logging.error("Unexpected response format in choices")
                break
        else:
            logging.error("No choices found in response")
            break

    return conversation_history

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

messages = [
    {"role": "system", "content": "You are a helpful assistant that can perform basic mathematical operations and get the current weather in a given location using the given tools."},
    {"role": "user", "content": "What's the weather like in Tokyo? and what is 2+2?"}
]

# Process the conversation with tool calling
conversation_history = process_conversation(messages, tools)

print("Final conversation:\n\n")

# Print the final conversation
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
