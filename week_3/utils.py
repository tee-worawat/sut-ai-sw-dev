"""Utility builders for week_3 agentic RAG notebooks (Gemini LLM + local embeds)."""

from typing import List, Optional

from llama_index.core import Settings, SummaryIndex, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import FilterCondition, MetadataFilters
from llama_index.llms.google_genai import GoogleGenAI

from helper import DEFAULT_LLM_MODEL, get_embed_model, get_google_api_key, load_pdf


def _default_llm():
    return GoogleGenAI(model=DEFAULT_LLM_MODEL, api_key=get_google_api_key())


def _default_embed_model():
    return get_embed_model()


def get_router_query_engine(file_path: str, llm=None, embed_model=None):
    """Build a router query engine over summary + vector indexes for a PDF."""
    llm = llm or _default_llm()
    embed_model = embed_model or _default_embed_model()

    documents = load_pdf(file_path)
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes, embed_model=embed_model)

    # use_async=False avoids bursting many parallel LLM calls into a low RPM quota
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=False,
        llm=llm,
    )
    vector_query_engine = vector_index.as_query_engine(llm=llm)

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description="Useful for summarization questions related to MetaGPT",
    )
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description="Useful for retrieving specific context from the MetaGPT paper.",
    )

    return RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=llm),
        query_engine_tools=[summary_tool, vector_tool],
        verbose=True,
        llm=llm,
    )


def get_doc_tools(file_path: str, name: str):
    """Return vector-query and summary tools for a single document."""
    documents = load_pdf(file_path)
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    vector_index = VectorStoreIndex(nodes)

    def vector_query(
        query: str,
        page_numbers: Optional[List[str]] = None,
    ) -> str:
        """Answer questions over a paper with optional page filters.

        Always leave page_numbers as None UNLESS there is a specific page
        you want to search for.

        Args:
            query: the string query to be embedded.
            page_numbers: Filter by set of pages. Leave as None to search
                all pages; otherwise filter by the specified pages.
        """
        page_numbers = page_numbers or []
        metadata_dicts = [
            {"key": "page_label", "value": p} for p in page_numbers
        ]
        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR,
            ),
        )
        return query_engine.query(query)

    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_{name}",
        fn=vector_query,
    )

    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=False,
    )
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name}",
        query_engine=summary_query_engine,
        description=(
            f"Use ONLY IF you want to get a holistic summary of {name}. "
            f"Do NOT use if you have specific questions over {name}."
        ),
    )

    return vector_query_tool, summary_tool


def ensure_settings():
    """Ensure global Settings are configured if not already set."""
    if Settings.llm is None or Settings.embed_model is None:
        from helper import configure_settings

        configure_settings()
