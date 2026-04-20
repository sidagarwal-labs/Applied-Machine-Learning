"""
features.py - Compute features for (question, chunk) pairs for ranking models.

Features include:
  - BM25 score (lexical relevance)
  - Cosine similarity (semantic relevance)  
  - Chunk metadata (word count, sentence count)
  - Question metadata (complexity, hop count)
  - Query/reasoning type encodings
  - Text overlap features
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_text_overlap(question, chunk_text):
    """
    Compute simple text overlap features between question and chunk.
    Returns dict of overlap features.
    """
    q_tokens = set(question.lower().split())
    c_tokens = set(chunk_text.lower().split())

    if len(q_tokens) == 0:
        return {'token_overlap_ratio': 0.0, 'token_overlap_count': 0}

    overlap = q_tokens & c_tokens
    return {
        'token_overlap_ratio': len(overlap) / len(q_tokens),
        'token_overlap_count': len(overlap),
    }


def compute_features(triples_df, bm25_index, tfidf_index):
    """
    Compute feature matrix for all (question, chunk) pairs.
    Uses pair-wise sparse dot products — no full score matrix needed.

    Args:
        triples_df: DataFrame with question, chunk_id, chunk_text, and metadata
        bm25_index: BM25Index instance
        tfidf_index: TfidfIndex instance

    Returns:
        DataFrame with all original columns plus computed features
    """
    import scipy.sparse as sp
    from sklearn.preprocessing import normalize

    n_pairs = len(triples_df)
    unique_questions = triples_df['question'].unique()
    print(f"Computing features for {n_pairs} pairs ({len(unique_questions)} unique questions)...")

    # Build question → index mapping
    q_to_idx = {q: i for i, q in enumerate(unique_questions)}

    # Map each pair to its question index and chunk index
    q_indices = triples_df['question'].map(q_to_idx).values
    c_indices = triples_df['chunk_id'].map(bm25_index.chunk_id_to_idx).values

    # --- BM25 scoring (pair-wise sparse dot products) ---
    print("BM25 scoring (pair-wise)...")
    # Build query IDF vectors for unique questions
    q_vecs_bm25 = bm25_index.vectorizer.transform(unique_questions)
    q_idf = q_vecs_bm25.multiply(bm25_index.idf).tocsr()  # (n_unique_q, V) CSR

    # Ensure adjusted_tf is CSR for row indexing
    adj_tf = sp.csr_matrix(bm25_index.adjusted_tf)

    # For each pair, compute: q_idf[q_idx] · adjusted_tf[c_idx]
    # Gather the rows we need: one per pair
    q_rows = q_idf[q_indices]          # (n_pairs, V) sparse
    c_rows = adj_tf[c_indices]         # (n_pairs, V) sparse
    # Element-wise multiply then sum each row → dot product per pair
    bm25_scores = np.array(q_rows.multiply(c_rows).sum(axis=1)).flatten()
    print(f"  BM25 scores computed: {bm25_scores.shape}")

    # --- TF-IDF cosine similarity (pair-wise sparse dot products) ---
    print("TF-IDF scoring (pair-wise)...")
    q_vecs_tfidf = tfidf_index.vectorizer.transform(unique_questions)
    # Normalize both query and chunk vectors for cosine similarity
    q_norm = normalize(q_vecs_tfidf, norm='l2')
    c_norm = normalize(sp.csr_matrix(tfidf_index.tfidf_matrix), norm='l2')

    q_rows_tfidf = q_norm[q_indices]   # (n_pairs, V) sparse
    c_rows_tfidf = c_norm[c_indices]   # (n_pairs, V) sparse
    cosine_sims = np.array(q_rows_tfidf.multiply(c_rows_tfidf).sum(axis=1)).flatten()
    print(f"  TF-IDF scores computed: {cosine_sims.shape}")

    # --- Compute text overlap features ---
    print("Computing text overlap features...")
    overlap_ratios = np.zeros(n_pairs)
    overlap_counts = np.zeros(n_pairs, dtype=int)
    q_lengths = np.zeros(n_pairs, dtype=int)

    for i, (_, row) in enumerate(triples_df.iterrows()):
        overlap = compute_text_overlap(row['question'], row['chunk_text'])
        overlap_ratios[i] = overlap['token_overlap_ratio']
        overlap_counts[i] = overlap['token_overlap_count']
        q_lengths[i] = len(row['question'].split())

    features_df = pd.DataFrame({
        'bm25_score': bm25_scores,
        'cosine_similarity': cosine_sims,
        'token_overlap_ratio': overlap_ratios,
        'token_overlap_count': overlap_counts,
        'question_length': q_lengths,
    })

    # One-hot encode query_type and reasoning_type
    query_type_dummies = pd.get_dummies(triples_df['query_type'], prefix='qtype')
    reasoning_type_dummies = pd.get_dummies(triples_df['reasoning_type'], prefix='rtype')

    # Combine everything
    result = pd.concat([
        triples_df.reset_index(drop=True),
        features_df.reset_index(drop=True),
        query_type_dummies.reset_index(drop=True),
        reasoning_type_dummies.reset_index(drop=True),
    ], axis=1)

    return result


def get_feature_columns(df):
    """
    Return the list of feature column names to use for model training.
    Excludes metadata/target columns.
    """
    exclude = {
        'question', 'chunk_id', 'chunk_text', 'relevance',
        'query_type', 'reasoning_type', 'doc_idx', 'segment_ids'
    }
    return [c for c in df.columns if c not in exclude]
