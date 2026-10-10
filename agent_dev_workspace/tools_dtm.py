from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

class SimpleToolWrapper:
    def __init__(self, name: str, func, description: str = ""):
        self.name = name
        self.func = func
        self.description = description

    def invoke(self, input_data: Any) -> Any:
        if isinstance(input_data, dict):
            return self.func(**input_data)
        return self.func(input_data)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def build_dtm_tool(
    session,
    max_features: Optional[int] = None,
    min_df: int = 1,
    ngram_range: List[int] = [1, 1],
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
    result_data = {
        "result_id": result_id,
        "dtm_matrix": dtm_matrix,
        "feature_names": feature_names,
        "vectorizer": vectorizer,
        "n_documents": n_docs,
        "n_terms": n_terms,
        "non_zero": non_zero,
        "total_elements": total_elements,
        "sparsity_pct": sparsity_pct,
    }

    session.feature_matrix = dtm_matrix
    session.feature_names = feature_names

    if hasattr(session, "artifacts") and isinstance(session.artifacts, dict):
        session.artifacts["count_vectorizer"] = vectorizer

    if hasattr(session, "results_store") and isinstance(session.results_store, dict):
        session.results_store[result_id] = result_data
    setattr(session, result_id, result_data)

    return {
        "result_id": result_id,
        "n_documents": n_docs,
        "n_terms": n_terms,
        "non_zero": non_zero,
        "total_elements": total_elements,
        "sparsity_pct": sparsity_pct,
    }


def term_frequency_tool(session, dtm_result_id: str = "build_dtm_1", top_n: int = 20) -> Dict[str, Any]:
    dtm_matrix = getattr(session, "feature_matrix", None)
    feature_names = getattr(session, "feature_names", None)

    if dtm_matrix is None or feature_names is None:
        raise ValueError(f"DTM result '{dtm_result_id}' not found in session.")

    counts = np.asarray(dtm_matrix.sum(axis=0)).ravel()
    frequencies = {term: int(count) for term, count in zip(feature_names, counts)}

    result_id = "term_frequency_1"
    result_data = {
        "result_id": result_id,
        "frequencies": frequencies,
        "total_terms": len(feature_names),
    }

    if hasattr(session, "results_store") and isinstance(session.results_store, dict):
        session.results_store[result_id] = result_data
    setattr(session, result_id, result_data)

    return result_data


def dtm_heatmap_tool(
    session,
    dtm_result_id: str = "build_dtm_1",
    n_documents: int = 20,
    n_terms: int = 20,
    n_docs: Optional[int] = None,
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
    slice_list = dense_slice.tolist()
    slice_features = feature_names[:slice_terms_count]

    # 繪製 Heatmap Figure 並存入 session.pending_figure
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(dense_slice, cmap="Blues")
    fig.colorbar(cax)
    ax.set_xticks(range(len(slice_features)))
    ax.set_xticklabels(slice_features, rotation=45, ha="left")
    session.pending_figure = fig

    result_id = "dtm_heatmap_1"
    result_data = {
        "result_id": result_id,
        "slice": slice_list,             # 對齊測試要求的 slice 欄位
        "matrix_slice": slice_list,      # 保留相容性
        "doc_indices": list(range(slice_docs_count)),
        "feature_names": slice_features,
        "n_documents": slice_docs_count,
        "n_terms": slice_terms_count,
    }

    if hasattr(session, "results_store") and isinstance(session.results_store, dict):
        session.results_store[result_id] = result_data
    setattr(session, result_id, result_data)

    return result_data


def make_tools(session):
    return [
        SimpleToolWrapper("build_dtm_tool", lambda **kw: build_dtm_tool(session, **kw)),
        SimpleToolWrapper("term_frequency_tool", lambda **kw: term_frequency_tool(session, **kw)),
        SimpleToolWrapper("dtm_heatmap_tool", lambda **kw: dtm_heatmap_tool(session, **kw)),
    ]
