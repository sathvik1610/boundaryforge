import os
import litellm
from litellm import completion

key = os.getenv("HUGGINGFACE_API_KEY", "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")

try:
    print("Testing native LiteLLM HuggingFace...")
    res = completion(
        model="huggingface/Qwen/Qwen2.5-72B-Instruct",
        messages=[{"role": "user", "content": "Hello"}],
        api_key=key,
        max_tokens=10
    )
    print("LiteLLM SUCCESS:", res.choices[0].message.content)
except Exception as e:
    print("LiteLLM failed:", e)

from openai import OpenAI
try:
    print("Testing OpenAI client with models endpoint...")
    client = OpenAI(base_url="https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct/v1", api_key=key)
    res = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    print("OpenAI SUCCESS:", res.choices[0].message.content)
except Exception as e:
    print("OpenAI failed:", e)
