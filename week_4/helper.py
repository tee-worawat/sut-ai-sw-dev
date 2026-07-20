"""Shared helpers for week_4 RAG notebooks (Gemini LLM + local embeds + triad eval)."""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import pandas as pd
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

DEFAULT_LLM_MODEL = "gemma-4-31b-it"
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


def get_eval_llm(model: str = DEFAULT_LLM_MODEL, temperature: float = 0.0):
    """Return a Gemini LLM configured for evaluation (deterministic judge)."""
    from llama_index.llms.google_genai import GoogleGenAI

    return GoogleGenAI(
        model=model,
        api_key=get_google_api_key(),
        temperature=temperature,
    )


def get_rag_triad_evaluators(llm=None):
    """Return Answer Relevancy, Context Relevancy, and Faithfulness evaluators."""
    from llama_index.core.evaluation import (
        AnswerRelevancyEvaluator,
        ContextRelevancyEvaluator,
        FaithfulnessEvaluator,
    )

    judge = llm or get_eval_llm()
    return {
        "answer_relevancy": AnswerRelevancyEvaluator(llm=judge),
        "context_relevancy": ContextRelevancyEvaluator(llm=judge),
        "groundedness": FaithfulnessEvaluator(llm=judge),
    }


def _contexts_from_response(response: Any) -> list[str]:
    source_nodes = getattr(response, "source_nodes", None) or []
    contexts: list[str] = []
    for node_with_score in source_nodes:
        node = getattr(node_with_score, "node", node_with_score)
        text = getattr(node, "get_content", None)
        if callable(text):
            contexts.append(text())
        else:
            contexts.append(getattr(node, "text", str(node)))
    return contexts


def _score_or_none(eval_result: Any) -> Optional[float]:
    score = getattr(eval_result, "score", None)
    if score is not None:
        return float(score)
    passing = getattr(eval_result, "passing", None)
    if passing is None:
        return None
    return 1.0 if passing else 0.0


def evaluate_query_engine(
    query_engine,
    questions: Sequence[str],
    app_id: str = "app",
    llm=None,
    include_feedback: bool = False,
) -> pd.DataFrame:
    """Run the RAG triad over a query engine and return a score DataFrame.

    Metrics (0–1, higher is better):
    - answer_relevancy: Answer Relevancy
    - context_relevancy: Context Relevancy
    - groundedness: Faithfulness / groundedness
    """
    evaluators = get_rag_triad_evaluators(llm=llm)
    rows: list[dict[str, Any]] = []

    for question in questions:
        response = query_engine.query(question)
        answer = str(response)
        contexts = _contexts_from_response(response)

        ar = evaluators["answer_relevancy"].evaluate(
            query=question,
            response=answer,
        )
        cr = evaluators["context_relevancy"].evaluate(
            query=question,
            contexts=contexts,
        )
        # Faithfulness / groundedness
        fa = evaluators["groundedness"].evaluate_response(response=response)

        row: dict[str, Any] = {
            "app_id": app_id,
            "question": question,
            "answer": answer,
            "answer_relevancy": _score_or_none(ar),
            "context_relevancy": _score_or_none(cr),
            "groundedness": _score_or_none(fa),
        }
        if include_feedback:
            row["answer_relevancy_feedback"] = getattr(ar, "feedback", None)
            row["context_relevancy_feedback"] = getattr(cr, "feedback", None)
            row["groundedness_feedback"] = getattr(fa, "feedback", None)
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_eval_df(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of the three triad metrics, grouped by app_id when present."""
    metric_cols = ["answer_relevancy", "context_relevancy", "groundedness"]
    available = [c for c in metric_cols if c in df.columns]
    if not available:
        return pd.DataFrame()
    if "app_id" in df.columns:
        return df.groupby("app_id", as_index=False)[available].mean(numeric_only=True)
    return df[available].mean(numeric_only=True).to_frame().T
