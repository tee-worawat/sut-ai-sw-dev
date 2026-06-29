"""Shared Gemini API helpers for week_1 notebooks."""

import os

from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types

_ = load_dotenv(find_dotenv())

DEFAULT_MODEL = "gemini-3.1-flash-lite"
_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set. Copy .env.example to .env and add your key "
                "from https://aistudio.google.com/apikey"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def get_completion(prompt, model=DEFAULT_MODEL, temperature=0):
    response = _get_client().models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )
    return response.text


def _messages_to_gemini_contents(messages):
    system_instruction = None
    contents = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            if system_instruction is None:
                system_instruction = content
            else:
                system_instruction = f"{system_instruction}\n{content}"
        elif role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=content)]))
        elif role == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part(text=content)]))
    return system_instruction, contents


def get_completion_from_messages(messages, model=DEFAULT_MODEL, temperature=0):
    system_instruction, contents = _messages_to_gemini_contents(messages)
    response = _get_client().models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        ),
    )
    return response.text
