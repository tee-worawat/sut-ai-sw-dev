#!pip install python-dotenv


import os
from dotenv import load_dotenv, find_dotenv

import nest_asyncio

nest_asyncio.apply()


def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")


def get_google_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("GOOGLE_API_KEY")


def get_hf_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("HUGGINGFACE_API_KEY")


# Re-export shared RAG triad eval helpers (Gemini judge via week_4/helper.py)
import sys
from pathlib import Path as _Path

_WEEK4 = _Path(__file__).resolve().parent.parent
if str(_WEEK4) not in sys.path:
    sys.path.insert(0, str(_WEEK4))

from helper import (  # noqa: E402
    get_eval_llm,
    get_rag_triad_evaluators,
    evaluate_query_engine,
    summarize_eval_df,
)

from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank


def build_sentence_window_index(
    document, llm, embed_model="local:BAAI/bge-small-en-v1.5", save_dir="sentence_index"
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    embed_model_obj = embed_model
    if isinstance(embed_model, str) and embed_model.startswith("local:"):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        embed_model_obj = HuggingFaceEmbedding(
            model_name=embed_model.split("local:", 1)[1]
        )

    Settings.llm = llm
    Settings.embed_model = embed_model_obj
    Settings.node_parser = node_parser

    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents([document])
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir)
        )

    return sentence_index


def get_sentence_window_query_engine(
    sentence_index,
    similarity_top_k=6,
    rerank_top_n=2,
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine


from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine


def build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index",
    chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)

    embed_model_obj = embed_model
    if isinstance(embed_model, str) and embed_model.startswith("local:"):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        embed_model_obj = HuggingFaceEmbedding(
            model_name=embed_model.split("local:", 1)[1]
        )

    Settings.llm = llm
    Settings.embed_model = embed_model_obj

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir)
        )
    return automerging_index


def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=6,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine
