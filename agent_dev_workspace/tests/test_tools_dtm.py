import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools

def setup_test_environment():
    """Helper to instantiate a fresh session and load premade + workspace DTM tools."""
    session = SessionState()
    
    # Load premade tools (load_dataset_tool) bound to session
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load tools_dtm tools bound to same session
    dtm_tools = make_tools(session)
    tools_dict = {t.name: t for t in dtm_tools}
    
    # Populate session using premade load_dataset_tool with sample fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    return session, tools_dict

def test_build_dtm_tool():
    """Test build_dtm_tool against Known Answer #4 in TEST_FIXTURE.md."""
    session, tools = setup_test_environment()
    build_dtm = tools["build_dtm_tool"]
    
    result = build_dtm.invoke({})
    
    # Check returns and vocabulary
    assert result["n_documents"] == 8
    assert result["n_terms"] == 5
    expected_vocab = ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_names == expected_vocab
    
    # Check matrix shape and non-zero entries
    assert session.feature_matrix.shape == (8, 5)
    assert session.feature_matrix.nnz == 21
    assert pytest.approx(result["sparsity_pct"], abs=1e-3) == 47.5
    
    # Check that fitted CountVectorizer is saved in session.artifacts
    assert "count_vectorizer" in session.artifacts

def test_term_frequency_tool():
    """Test term_frequency_tool against Known Answer #14 in TEST_FIXTURE.md."""
    session, tools = setup_test_environment()
    build_dtm = tools["build_dtm_tool"]
    term_freq = tools["term_frequency_tool"]
    
    # Build DTM first
    build_dtm.invoke({})
    
    # Run term_frequency_tool
    result = term_freq.invoke({})
    
    expected_frequencies = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7
    }
    
    # Verify frequencies returned in summary
    frequencies = result.get("frequencies", {})
    for term, expected_count in expected_frequencies.items():
        assert frequencies[term] == expected_count

def test_dtm_heatmap_tool():
    """Test dtm_heatmap_tool against Known Answer #21 in TEST_FIXTURE.md."""
    session, tools = setup_test_environment()
    build_dtm = tools["build_dtm_tool"]
    dtm_heatmap = tools["dtm_heatmap_tool"]
    
    # Build DTM first
    build_dtm.invoke({})
    
    # 1. Full matrix slice (defaults n_terms=20, n_documents=20)
    full_res = dtm_heatmap.invoke({})
    assert full_res["n_documents"] == 8
    assert full_res["n_terms"] == 5
    assert session.pending_figure is not None
    
    # 2. Slice corner (n_terms=2, n_documents=3)
    corner_res = dtm_heatmap.invoke({"n_terms": 2, "n_documents": 3})
    assert corner_res["n_documents"] == 3
    assert corner_res["n_terms"] == 2
    assert corner_res["slice"] == [[3, 1], [2, 1], [1, 1]]
