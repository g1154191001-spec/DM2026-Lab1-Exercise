from typing import Any, Dict, List, Optional
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import CountVectorizer

# Define Pydantic schemas for LangChain schema validation
class BuildDTMInput(BaseModel):
    max_features: Optional[int] = Field(None, description="Maximum number of features")
    min_df: int = Field(1, description="Minimum document frequency")
    ngram_range: List[int] = Field([1, 1], description="N-gram range tuple [min, max]")

class TermFrequencyInput(BaseModel):
    dtm_result_id: str = Field("build_dtm_1", description="Result ID of the DTM to analyze")
    top_n: int = Field(20, description="Number of top frequent terms to return")

class DTMHeatmapInput(BaseModel):
    dtm_result_id: str = Field("build_dtm_1", description="Result ID of the DTM")
    n_documents: int = Field(20, description="Number of documents to display in heatmap")
    n_terms: int = Field(20, description="Number of terms to display in heatmap")


class SimpleToolWrapper:
    def __init__(self, name: str, func, description: str = "", args_schema: Optional[type[BaseModel]] = None):
        self.name = name
        self.func = func
        self.description = description
        self.args_schema = args_schema

    def invoke(self, input_data: Any, **kwargs) -> Any:
        if isinstance(input_data, BaseModel):
            combined_args = input_data.model_dump()
        elif isinstance(input_data, dict):
            combined_args = {**input_data, **kwargs}
        else:
            combined_args = dict(kwargs)

        combined_args.pop("kwargs", None)
        combined_args.pop("session", None)

        sig = inspect.signature(self.func)
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

        if has_var_keyword:
            valid_args = combined_args
        else:
            valid_args = {k: v for k, v in combined_args.items() if k in sig.parameters}

        return self.func(**valid_args)

    def __call__(self, *args, **kwargs):
        return self.invoke(kwargs)


class RecordAdapter:
    """符合 agent_pipeline/grounding.py 規範的 Record 結構"""
    def __init__(self, summary: dict):
        self.summary = summary


def _register_result(session, result_id: str, summary: dict, full_data: dict):
    """遵照管道框架格式將結果寫入 session"""
    if hasattr(session, "results_store") and isinstance(session.results_store, dict):
        session.results_store[result_id] = {
            "record": RecordAdapter(summary),
            "data": full_data
        }
    setattr(session, result_id, full_data)


def build_dtm_tool(
    session,
    max_features: Optional[int] = None,
    min_df: int = 1,
    ngram_range: List[int] = [1, 1],
    **kwargs,
) -> Dict[str, Any]:
    df = getattr(session, "dataframe", None)
    if df is None:
        raw = getattr(session, "raw_dataset", None)
        if isinstance(raw, dict) and "df" in raw:
            df = raw["df"]
        elif hasattr(raw, "text"):
            df = raw

    if df is None:
        raise ValueError("No dataset loaded in session. Run load_dataset_tool first.")

    text_col = "text" if "text" in df.columns else df.select_dtypes(include=["object", "string"]).columns[0]
    texts = df[text_col].astype(str).tolist()

    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=tuple(ngram_range),
    )
    dtm_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out().tolist()

    n_docs, n_terms = dtm_matrix.shape
    non_zero = int(dtm_matrix.nnz)
    total_elements = n_docs * n_terms
    sparsity_pct = float((1.0 - (non_zero / total_elements)) * 100.0) if total_elements > 0 else 0.0

    result_id = "build_dtm_1"
    summary = {
        "result_id": result_id,
        "n_documents": n_docs,
        "n_terms": n_terms,
        "non_zero": non_zero,
        "total_elements": total_elements,
        "sparsity_pct": sparsity_pct,
    }
    full_data = {
        **summary,
        "dtm_matrix": dtm_matrix,
        "feature_names": feature_names,
        "vectorizer": vectorizer,
    }

    session.feature_matrix = dtm_matrix
    session.feature_names = feature_names

    if hasattr(session, "artifacts") and isinstance(session.artifacts, dict):
        session.artifacts["count_vectorizer"] = vectorizer

    _register_result(session, result_id, summary, full_data)

    return summary


def term_frequency_tool(
    session,
    dtm_result_id: str = "build_dtm_1",
    top_n: int = 20,
    **kwargs,
) -> Dict[str, Any]:
    dtm_matrix = getattr(session, "feature_matrix", None)
    feature_names = getattr(session, "feature_names", None)

    if dtm_matrix is None or feature_names is None:
        raise ValueError(f"DTM result '{dtm_result_id}' not found in session.")

    counts = np.asarray(dtm_matrix.sum(axis=0)).ravel()
    sorted_indices = np.argsort(counts)[::-1]
    top_indices = sorted_indices[:top_n]
    frequencies = {feature_names[i]: int(counts[i]) for i in top_indices}

    result_id = "term_frequency_1"
    summary = {
        "result_id": result_id,
        "total_terms": len(feature_names),
        "frequencies": frequencies,
    }
    
    _register_result(session, result_id, summary, summary)

    return summary


def dtm_heatmap_tool(
    session,
    dtm_result_id: str = "build_dtm_1",
    n_documents: int = 20,
    n_terms: int = 20,
    n_docs: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    num_docs = n_docs if n_docs is not None else n_documents

    dtm_matrix = getattr(session, "feature_matrix", None)
    feature_names = getattr(session, "feature_names", None)

    if dtm_matrix is None or feature_names is None:
        raise ValueError(f"DTM result '{dtm_result_id}' not found in session.")

    actual_n_docs, actual_n_terms = dtm_matrix.shape

    slice_docs_count = min(num_docs, actual_n_docs)
    slice_terms_count = min(n_terms, actual_n_terms)

    dense_slice = dtm_matrix[:slice_docs_count, :slice_terms_count].toarray()
    slice_features = feature_names[:slice_terms_count]

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(dense_slice, cmap="Blues")
    fig.colorbar(cax)
    ax.set_xticks(range(len(slice_features)))
    ax.set_xticklabels(slice_features, rotation=45, ha="left")
    session.pending_figure = fig

    slice_list = dense_slice.tolist()

    result_id = "dtm_heatmap_1"
    summary = {
        "result_id": result_id,
        "n_documents": slice_docs_count,
        "n_terms": slice_terms_count,
    }
    full_data = {
        **summary,
        "slice": slice_list,
        "matrix_slice": slice_list,
        "doc_indices": list(range(slice_docs_count)),
        "feature_names": slice_features,
    }

    _register_result(session, result_id, summary, full_data)

    return summary


def make_tools(session):
    return [
        SimpleToolWrapper(
            "build_dtm_tool",
            lambda **kw: build_dtm_tool(session, **kw),
            "Builds a Document-Term Matrix (DTM)",
            args_schema=BuildDTMInput
        ),
        SimpleToolWrapper(
            "term_frequency_tool",
            lambda **kw: term_frequency_tool(session, **kw),
            "Calculates term frequencies across all documents",
            args_schema=TermFrequencyInput
        ),
        SimpleToolWrapper(
            "dtm_heatmap_tool",
            lambda **kw: dtm_heatmap_tool(session, **kw),
            "Generates a heatmap slice of the DTM",
            args_schema=DTMHeatmapInput
        ),
    ]
