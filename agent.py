# agent.py
import json
import os
import httpx

from tools import calculator, get_weather, web_search

# System instructions for your AI Study Assistant
SYSTEM_INSTRUCTIONS = """
You are an AI Study Assistant.
Help the user with Python, programming, mathematics, Agentic AI, weather, current information, and general knowledge.

You have three tools available:
1. calculate(operation, a, b): Use for mathematical calculations.
2. get_weather_tool(city): Use for current weather questions.
3. web_search_tool(question): Use when the answer requires current, latest, or real-time web information.

IMPORTANT STRUCTURED OUTPUT REQUIREMENT:
When you provide your final response to the user, you MUST respond ONLY with a valid JSON object (no markdown code blocks, just raw JSON) containing these exact keys:
{
    "answer": "Your detailed answer here",
    "topic": "The main topic (e.g., Python, Weather, Math)",
    "difficulty": "Beginner, Intermediate, or Advanced",
    "used_calculator": true or false,
    "used_weather": true or false,
    "used_web_search": true or false
}
"""

# Define tool schemas for OpenRouter/OpenAI tool calling
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "description": "add, subtract, multiply, divide, +, -, *, /"},
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["operation", "a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_tool",
            "description": "Get current weather for any city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Name of the city"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search_tool",
            "description": "Search the web for current or up-to-date information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Search query"}
                },
                "required": ["question"]
            }
        }
    }
]


def run_agent(user_question: str, message_history: list) -> tuple:
    """Sends the conversation to OpenRouter, executes tools if requested, and returns parsed output + history."""
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    url = "https://openrouter.ai/api/v1/chat/completions"

    # Append user question to history
    message_history.append({"role": "user", "content": user_question})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    while True:
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM_INSTRUCTIONS}] + message_history,
            "tools": TOOLS_SCHEMA,
            "tool_choice": "auto",
        }

        response = httpx.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]

        # Check if the model wants to call tools
        if message.get("tool_calls"):
            message_history.append(message)

            for tool_call in message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                tool_call_id = tool_call["id"]


                tool_result = None
                try:
                    if tool_name == "calculate":
                        tool_result = calculator(tool_args["operation"], tool_args["a"], tool_args["b"])
                    elif tool_name == "get_weather_tool":
                        tool_result = get_weather(tool_args["city"])
                    elif tool_name == "web_search_tool":
                        tool_result = web_search(tool_args["question"])
                    else:
                        tool_result = f"Error: Unknown tool {tool_name}"
                except Exception as e:
                    tool_result = f"Tool error: {str(e)}"

                message_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(tool_result)
                })

            continue  # Loop back so LLM can read tool results

        else:
            # Final response received
            assistant_content = message.get("content", "")
            message_history.append({"role": "assistant", "content": assistant_content})

            # Clean markdown code blocks if the LLM wraps the JSON
            cleaned_json_str = assistant_content.strip()
            if cleaned_json_str.startswith("```json"):
                cleaned_json_str = cleaned_json_str[7:]
            if cleaned_json_str.startswith("```"):
                cleaned_json_str = cleaned_json_str[3:]
            if cleaned_json_str.endswith("```"):
                cleaned_json_str = cleaned_json_str[:-3]
            cleaned_json_str = cleaned_json_str.strip()

            try:
                output_data = json.loads(cleaned_json_str)
                return output_data, message_history
            except json.JSONDecodeError:
                # Fallback dictionary if parsing fails
                fallback_output = {
                    "answer": assistant_content,
                    "topic": "General",
                    "difficulty": "Beginner",
                    "used_calculator": False,
                    "used_weather": False,
                    "used_web_search": False
                }
                return fallback_output, message_history