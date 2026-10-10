from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from agent_pipeline.tool_registry import register_tool, SimpleTool


@register_tool("build_dtm_tool")
def build_dtm_tool(
    session,
    max_features: Optional[int] = None,
    min_df: int = 1,
    ngram_range: List[int] = [1, 1],
) -> Dict[str, Any]:
    """Builds a Document-Term Matrix (DTM) from loaded dataset text."""
    if "loaded_data" not in session:
        raise ValueError("No dataset loaded in session. Run load_dataset_tool first.")

    df = session["loaded_data"]["df"]
    text_col = session["loaded_data"]["text_column"]
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

    session["build_dtm_tool"] = result_data
    session[result_id] = result_data

    return {
        "result_id": result_id,
        "n_documents": n_docs,
        "n_terms": n_terms,
        "non_zero": non_zero,
        "total_elements": total_elements,
        "sparsity_pct": sparsity_pct,
    }


@register_tool("term_frequency_tool")
def term_frequency_tool(session, dtm_result_id: str = "build_dtm_1", top_n: int = 20) -> Dict[str, Any]:
    """Computes total term frequencies across all documents in the DTM."""
    if dtm_result_id not in session:
        raise ValueError(f"DTM result '{dtm_result_id}' not found in session.")

    dtm_data = session[dtm_result_id]
    dtm_matrix = dtm_data["dtm_matrix"]
    feature_names = dtm_data["feature_names"]

    counts = np.asarray(dtm_matrix.sum(axis=0)).ravel()
    freq_df = pd.DataFrame({"term": feature_names, "frequency": counts})
    freq_df = freq_df.sort_values(by="frequency", ascending=False).reset_index(drop=True)

    top_terms = freq_df.head(top_n).to_dict(orient="records")

    result_id = f"term_frequency_{len(session) + 1}"
    result_data = {
        "result_id": result_id,
        "top_n": top_n,
        "top_terms": top_terms,
        "total_terms": len(feature_names),
    }

    session["term_frequency_tool"] = result_data
    session[result_id] = result_data

    return result_data


@register_tool("dtm_heatmap_tool")
def dtm_heatmap_tool(
    session,
    dtm_result_id: str = "build_dtm_1",
    n_docs: int = 10,
    n_terms: int = 10,
) -> Dict[str, Any]:
    """Extracts a slice of the DTM matrix for heatmap visualization."""
    if dtm_result_id not in session:
        raise ValueError(f"DTM result '{dtm_result_id}' not found in session.")

    dtm_data = session[dtm_result_id]
    dtm_matrix = dtm_data["dtm_matrix"]
    feature_names = dtm_data["feature_names"]

    slice_matrix = dtm_matrix[:n_docs, :n_terms].toarray().tolist()
    slice_features = feature_names[:n_terms]

    result_id = f"dtm_heatmap_{len(session) + 1}"
    result_data = {
        "result_id": result_id,
        "matrix_slice": slice_matrix,
        "doc_indices": list(range(n_docs)),
        "feature_names": slice_features,
    }

    session["dtm_heatmap_tool"] = result_data
    session[result_id] = result_data

    return result_data


def make_tools(session):
    """Factory function returning a list of bound Tool objects."""
    return [
        SimpleTool(
            name="build_dtm_tool",
            description="Builds DTM matrix",
            func=lambda **kw: build_dtm_tool(session, **kw),
        ),
        SimpleTool(
            name="term_frequency_tool",
            description="Computes term frequency",
            func=lambda **kw: term_frequency_tool(session, **kw),
        ),
        SimpleTool(
            name="dtm_heatmap_tool",
            description="Extracts heatmap slice",
            func=lambda **kw: dtm_heatmap_tool(session, **kw),
        ),
    ]
