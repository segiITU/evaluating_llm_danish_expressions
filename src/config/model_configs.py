import os
from typing import Dict, Any
from openai import OpenAI
import google.generativeai as genai
from llamaapi import LlamaAPI
import anthropic

API_KEYS = {
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
    "LLAMA_API_KEY": os.getenv("LLAMA_API_KEY", ""),
    "XAI_API_KEY": os.getenv("XAI_API_KEY", ""),
    "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY", "")
}

MODEL_CONFIGS = {
    "llama-3.1": {
        "model_name": "llama-3.1-70b",
        "max_token": 1,
        "temperature": 0
    },
    "gpt-3.5-turbo": {
    "model_name": "gpt-3.5-turbo-0125",
    "max_tokens": 1,
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
    },
    "claude": {
        "model_name": "claude-3-sonnet-latest",
        "max_tokens": 1,
        "temperature": 0
    },
    "claude-3-5-sonnet": {
        "model_name": "claude-3-5-sonnet-20241022",
        "max_tokens": 1,
        "temperature": 0
    },
    "claude-3-7-sonnet": {
        "model_name": "claude-3-7-sonnet-20250219",
        "max_tokens": 1,
        "temperature": 0
    },
        "grok-2": {
        "model_name": "grok-2-1212",
        "max_tokens": 1,
        "temperature": 0
    },
      "deepseek": {  
        "model_name": "deepseek-chat", # points to DeepSeek-V3
        "max_tokens": 1,
        "temperature": 0
    }
}

PROMPT_TEMPLATE = """Betyder udtrykket "{idiom}" følgende: "{definition}"? Svar kun med ja eler nej."""