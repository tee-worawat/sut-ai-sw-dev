"""Shared setup helpers for week_3 notebooks (Gemini LLM + local embeddings)."""

import os

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

DEFAULT_LLM_MODEL = "gemma-4-31b-it"
# Local, free embedding model (no API quota). First run downloads ~33MB.
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def get_google_api_key() -> str:
    """Return GOOGLE_API_KEY from the environment (loaded from .env)."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not set. Copy .env.example to .env and add your key "
            "from https://aistudio.google.com/apikey"
        )
    return api_key


def get_embed_model(model_name: str = DEFAULT_EMBED_MODEL):
    """Return a local HuggingFace embedding model (runs on CPU, no API key)."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(model_name=model_name)


def load_pdf(file_path: str):
    """Load a PDF with text extraction (not raw binary).

    Plain SimpleDirectoryReader falls back to reading the file as text, which
    turns a PDF into thousands of junk chunks and blows rate limits on
    tree_summarize. Always use PDFReader for .pdf files.
    """
    from llama_index.core import SimpleDirectoryReader
    from llama_index.readers.file import PDFReader

    return SimpleDirectoryReader(
        input_files=[file_path],
        file_extractor={".pdf": PDFReader()},
    ).load_data()


def configure_settings(
    llm_model: str = DEFAULT_LLM_MODEL,
    embed_model: str = DEFAULT_EMBED_MODEL,
):
    """Configure LlamaIndex Settings: Gemini for LLM, local HF for embeddings."""
    from llama_index.core import Settings
    from llama_index.llms.google_genai import GoogleGenAI

    api_key = get_google_api_key()
    Settings.llm = GoogleGenAI(model=llm_model, api_key=api_key)
    Settings.embed_model = get_embed_model(embed_model)
    return Settings.llm, Settings.embed_model
