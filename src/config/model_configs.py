import os
from typing import Dict, Any
from openai import OpenAI
import google.generativeai as genai
from llamaapi import LlamaAPI

API_KEYS = {
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
    "LLAMA_API_KEY": os.getenv("LLAMA_API_KEY", "")
}

MODEL_CONFIGS = {
    "llama-3.1": {
        "model_name": "llama3.1-70b",
        "max_token": 1,
        "temperature": 0
    },
    "gpt-4": {
        "model_name": "gpt-4-0125-preview",
        "max_tokens": 1,
        "temperature": 0
    },
    "gpt-4o": {
        "model_name": "gpt-4o-2024-11-20",
        "max_tokens": 1,
        "temperature": 0
    },
    "gemini": {
        "model_name": "gemini-1.5-pro-latest",
        "temperature": 0,
        "max_output_tokens": 1
    }
}

PROMPT_TEMPLATE = """Choose the correct definition for the given metaphorical expression by responding with only a single letter representing your choice (A, B, C, or D).
Sentence: {metaphorical_expression}
Option A: {definition_a}
Option B: {definition_b}
Option C: {definition_c}
Option D: {definition_d}
Your response should be exactly one letter: A, B, C, or D."""

def validate_api_keys():
    missing_keys = [key for key, value in API_KEYS.items() if not value]
    if missing_keys:
        raise ValueError(f"Missing API keys: {', '.join(missing_keys)}")


def list_openai_models():
    client = OpenAI()
    models = client.models.list()
    for model in models.data:
        print(model.id)

def list_google_models():
    for model in genai.list_models():
        print(model.name)


#list_openai_models()
#list_google_models()
# 
# 