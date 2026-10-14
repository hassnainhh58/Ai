import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("LLM_MODEL")

API_URL = "https://openrouter.ai/api/v1/chat/completions"


def chat_with_ai(message: str) -> str:

    if not API_KEY or not MODEL:
        raise ValueError(
            "OPENROUTER_API_KEY or LLM_MODEL is missing from .env"
        )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": message,
            }
        ],
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    return data["choices"][0]["message"]["content"]


def main():
    print("===== Simple Grok Chat =====")

    while True:
        user_message = input("You: ").strip()

        if user_message.lower() == "exit":
            print("Goodbye!")
            break

        if not user_message:
            continue

        try:
            answer = chat_with_ai(user_message)
            print(f"\nAI: {answer}\n")


        except Exception as error:
            print(f"Error: {error}\n")


if __name__ == "__main__":
    main()