# Agentic Pipeline Analysis Report

**Author:** Hu Chia Yu  
**Student ID:** G1154191001  
**Date:** October 10, 2026  

---

## 1. Executive Summary

This report documents the execution and empirical analysis of an agent-driven text mining pipeline applied to the 20 Newsgroups dataset. The workflow was coordinated through an agent pipeline enforcing strict structural data handling (`RecordAdapter`) and batch authorization gate controls ("Gate 27"). The analysis pipeline performed document loading, Document-Term Matrix (DTM) construction, term frequency distribution analysis, and matrix slice heatmap visualization.

---

## 2. Experimental Pipeline & Results

### Step 1: Dataset Loading
- **Result ID:** `load_dataset_2`
- **Configuration:** 4 selected newsgroup categories (`alt.atheism`, `soc.religion.christian`, `comp.graphics`, `sci.med`).
- **Findings:** A total of **2,257 documents** were ingested into the session state. The category breakdown is as follows:
  - `soc.religion.christian`: 599 documents
  - `sci.med`: 594 documents
  - `comp.graphics`: 584 documents
  - `alt.atheism`: 480 documents

### Step 2: Document-Term Matrix Construction
- **Result ID:** `build_dtm_1`
- **Configuration:** `CountVectorizer` with default parameters (`ngram_range=(1, 1)`, `min_df=1`).
- **Findings:**
  - **Total Unique Terms:** 35,788
  - **Non-zero Entries:** 365,886
  - **Total Matrix Capacity:** 80,773,516
  - **Sparsity:** **99.55%**
- **Interpretation:** High sparsity (99.55%) is typical for un-stemmed, unfiltered raw text bag-of-words representations across diverse topic domains.

### Step 3: Term Frequency Analysis
- **Result ID:** `term_frequency_1`
- **Findings:** Extraction of the top 20 most frequent terms revealed high counts for unstopped English grammatical tokens:
  1. `the`: 29,825
  2. `of`: 17,660
  3. `to`: 17,222
  4. `and`: 13,017
  5. `is`: 12,343
  - Domain-specific structural tokens such as `edu` (rank 16, count 4,675) also appeared prominently due to email headers in the newsgroup format.

### Step 4: DTM Slice Heatmap Visualization
- **Result ID:** `dtm_heatmap_1`
- **Configuration:** Sub-matrix slice of `n_documents = 10` by `n_terms = 10`.
- **Findings & Visualization:** The sub-matrix heatmap demonstrates the extreme local sparsity of the raw text matrix.

![DTM Slice Heatmap](../plots/Hu_Chia_Yu_agentic-pipeline_plot_variance_1.png)

---

## 3. Discussion & Key Takeaways

1. **Gate Authorization & Security Integrity:** The execution verified that student-written analytical tools (`build_dtm_tool`, `term_frequency_tool`, `dtm_heatmap_tool`) properly pause at authorization gates until explicit human-in-the-loop batch authorization is provided.
2. **Data Model Compatibility:** Alignment with the pipeline's `grounding.py` requirements was achieved by wrapping outputs inside a structured `RecordAdapter` object containing a `.summary` attribute, preventing execution runtime key errors.
3. **Analytical Insight:** Without stop-word filtering or token preprocessing, common English articles and top-level domain extensions (`edu`) dominate the vocabulary list. Future iterations should incorporate `stop_words='english'` or TF-IDF weighting to highlight domain-distinguishing vocabulary across categories.
