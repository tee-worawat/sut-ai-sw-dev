"""Shared Gemini API helpers for week_2 notebooks."""

import json
import os
from types import SimpleNamespace
from typing import Any

from dotenv import find_dotenv, load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

_ = load_dotenv(find_dotenv())

DEFAULT_MODEL = "gemini-3.1-flash-lite"
_client = None


def _get_client() -> genai.Client:
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


def call_llm(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0) -> str:
    """Send a single-turn prompt and return the model's text response."""
    response = _get_client().models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )
    return response.text


def get_structured_output(
    prompt: str,
    response_model: type[BaseModel],
    model: str = DEFAULT_MODEL,
    temperature: float = 0,
    system_instruction: str | None = None,
) -> BaseModel:
    """Return a validated Pydantic model from Gemini structured JSON output."""
    response = _get_client().models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_json_schema=response_model.model_json_schema(),
        ),
    )
    return response_model.model_validate_json(response.text)


def get_structured_output_json(
    prompt: str,
    response_model: type[BaseModel],
    model: str = DEFAULT_MODEL,
    temperature: float = 0,
    system_instruction: str | None = None,
) -> tuple[str, Any]:
    """Return raw JSON text and the full Gemini response object."""
    response = _get_client().models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_json_schema=response_model.model_json_schema(),
        ),
    )
    return response.text, response


def tool_definitions_to_gemini_tools(
    tool_definitions: list[dict[str, Any]],
) -> list[types.Tool]:
    """Convert OpenAI-style tool definitions to Gemini Tool objects."""
    declarations = [
        types.FunctionDeclaration(
            name=tool["function"]["name"],
            description=tool["function"]["description"],
            parameters_json_schema=tool["function"]["parameters"],
        )
        for tool in tool_definitions
    ]
    return [types.Tool(function_declarations=declarations)]


def wrap_tool_calls(function_calls: list[types.FunctionCall]) -> list[SimpleNamespace]:
    """Wrap Gemini function calls for notebook code that expects OpenAI-style objects."""
    wrapped = []
    for function_call in function_calls:
        wrapped.append(
            SimpleNamespace(
                id=function_call.id,
                function=SimpleNamespace(
                    name=function_call.name,
                    arguments=json.dumps(function_call.args),
                ),
            )
        )
    return wrapped


def call_with_tools(
    messages: list[dict[str, str]],
    tool_definitions: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
) -> tuple[SimpleNamespace, list[SimpleNamespace], list[dict[str, str]]]:
    """Call Gemini with function tools and return an OpenAI-compatible message shape."""
    system_instruction = None
    user_content = None
    for message in messages:
        if message["role"] == "system":
            system_instruction = message["content"]
        elif message["role"] == "user":
            user_content = message["content"]

    response = _get_client().models.generate_content(
        model=model,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tool_definitions_to_gemini_tools(tool_definitions),
        ),
    )

    message = SimpleNamespace(
        content=response.text or "",
        tool_calls=wrap_tool_calls(response.function_calls or []),
    )
    return message, message.tool_calls, messages
