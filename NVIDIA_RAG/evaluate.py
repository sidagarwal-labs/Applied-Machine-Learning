"""evaluate.py - Ranking evaluation metrics (NDCG, Recall, MRR, MAP)."""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def dcg_at_k(relevances, k):
    """Compute DCG@k."""
    relevances = np.array(relevances[:k])
    if len(relevances) == 0:
        return 0.0
    discounts = np.log2(np.arange(len(relevances)) + 2)
    return np.sum(relevances / discounts)


def ndcg_at_k(relevances, k):
    """Compute NDCG@k."""
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def recall_at_k(relevances, k, total_relevant):
    """Compute Recall@k."""
    if total_relevant == 0:
        return 0.0
    return sum(relevances[:k]) / total_relevant


def reciprocal_rank(relevances):
    """Compute Reciprocal Rank."""
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(relevances):
    """Compute Average Precision."""
    if sum(relevances) == 0:
        return 0.0
    precisions = []
    relevant_count = 0
    for i, rel in enumerate(relevances):
        if rel > 0:
            relevant_count += 1
            precisions.append(relevant_count / (i + 1))
    return sum(precisions) / sum(relevances)


def evaluate_ranking(df, score_col, k_values=[5, 10, 20]):
    """Evaluate ranking model, returns dict of metric_name -> value."""
    results = {f'NDCG@{k}': [] for k in k_values}
    results.update({f'Recall@{k}': [] for k in k_values})
    results['MRR'] = []
    results['MAP'] = []

    for question, group in df.groupby('question'):
        sorted_group = group.sort_values(score_col, ascending=False)
        relevances = sorted_group['relevance'].tolist()
        total_relevant = sum(relevances)

        for k in k_values:
            results[f'NDCG@{k}'].append(ndcg_at_k(relevances, k))
            results[f'Recall@{k}'].append(recall_at_k(relevances, k, total_relevant))

        results['MRR'].append(reciprocal_rank(relevances))
        results['MAP'].append(average_precision(relevances))

    return {metric: np.mean(values) for metric, values in results.items()}


def prepare_candidates(qa_list, bm25_index, tfidf_index, chunk_lookup,
                       feature_cols, candidate_k=100):
    """Retrieve candidates and compute features once per question.

    Returns a list of dicts, one per question, each containing:
        - candidate_ids: list of chunk IDs
        - features: np.ndarray of shape (n_candidates, n_features)
        - golden_ids: set of relevant chunk IDs
        - metadata: dict with query_type, reasoning_type, counts
    """
    from features import compute_text_overlap
    from tqdm import tqdm

    #batch-encode all unique questions at once
    questions = [qa['question'] for qa in qa_list]
    unique_questions = list(dict.fromkeys(questions))
    q_to_idx = {q: i for i, q in enumerate(unique_questions)}

    print(f"  Batch BM25 scoring {len(unique_questions)} unique questions...")
    bm25_q_vecs = bm25_index.vectorizer.transform(unique_questions)
    bm25_q_idf = bm25_q_vecs.multiply(bm25_index.idf).tocsr()

    print(f"  Batch TF-IDF scoring {len(unique_questions)} unique questions...")
    tfidf_q_vecs = tfidf_index.vectorizer.transform(unique_questions)

    #batch-compute ALL score matrices at once (sparse × sparse, very fast)
    print("  Computing BM25 score matrix (batch)...")
    bm25_score_matrix = bm25_q_idf.dot(bm25_index.adjusted_tf.T).tocsr()
    print("  Computing TF-IDF score matrix (batch)...")
    tfidf_score_matrix = (tfidf_q_vecs @ tfidf_index.tfidf_matrix.T).tocsr()

    #precompute chunk_id -> index for fast lookup
    bm25_id_to_idx = bm25_index.chunk_id_to_idx
    tfidf_id_to_idx = tfidf_index.chunk_id_to_idx

    precomputed = []
    for qa in tqdm(qa_list, desc="Preparing candidates"):
        question = qa['question']
        golden_ids = set(qa['segment_ids'])
        qi = q_to_idx[question]

        #look up pre-computed score rows (no per-question matrix ops)
        bm25_all = bm25_score_matrix[qi].toarray().flatten()
        tfidf_all = tfidf_score_matrix[qi].toarray().flatten()

        #top-k from each
        bm25_topk = np.argsort(bm25_all)[-candidate_k:][::-1]
        tfidf_topk = np.argsort(tfidf_all)[-candidate_k:][::-1]

        seen = set()
        candidates = []
        for idx in list(bm25_topk) + list(tfidf_topk):
            cid = bm25_index.chunk_ids[idx]
            if cid not in seen:
                seen.add(cid)
                candidates.append(cid)

        #build feature rows using already-computed scores
        q_tokens = set(question.lower().split())
        q_len = len(question.split())
        rows = []
        for chunk_id in candidates:
            chunk = chunk_lookup[chunk_id]
            c_tokens = set(chunk['chunk_text'].lower().split())
            overlap_count = len(q_tokens & c_tokens)

            rows.append({
                'bm25_score': float(bm25_all[bm25_id_to_idx[chunk_id]]),
                'cosine_similarity': float(tfidf_all[tfidf_id_to_idx[chunk_id]]),
                'token_overlap_ratio': overlap_count / len(q_tokens) if q_tokens else 0.0,
                'token_overlap_count': overlap_count,
                'question_length': q_len,
                'chunk_word_count': chunk['word_count'],
                'chunk_sentence_count': chunk['sentence_count'],
                'question_complexity': qa['question_complexity'],
                'hop_count': qa['hop_count'],
            })

        feat_df = pd.DataFrame(rows)
        for col in feature_cols:
            if col not in feat_df.columns:
                feat_df[col] = 0

        precomputed.append({
            'candidate_ids': candidates,
            'features': feat_df[feature_cols].values,
            'bm25_scores': feat_df['bm25_score'].values,
            'tfidf_scores': feat_df['cosine_similarity'].values,
            'golden_ids': golden_ids,
            'metadata': {
                'question': question,
                'query_type': qa['query_type'],
                'reasoning_type': qa['reasoning_type'],
                'n_candidates': len(candidates),
                'n_golden': len(golden_ids),
                'golden_in_candidates': len(golden_ids & seen),
            },
        })

    return precomputed


def evaluate_from_candidates(precomputed, model, k_values=[5, 10, 20], show_progress=True):
    """Score precomputed candidates with a model and compute metrics.

    model=None -> BM25 baseline, model='tfidf' -> TF-IDF baseline,
    otherwise uses model.predict_proba or model.predict.
    """
    from tqdm.auto import tqdm

    results = {f'NDCG@{k}': [] for k in k_values}
    results.update({f'Recall@{k}': [] for k in k_values})
    results['MRR'] = []
    results['MAP'] = []
    meta_rows = []

    iterator = tqdm(precomputed, desc="Evaluating", disable=not show_progress)

    #batch model predictions across all questions for speed
    batched_scores = None
    if model is not None and model != 'tfidf':
        sizes = [len(e['candidate_ids']) for e in precomputed]
        all_features = np.vstack([e['features'] for e in precomputed])
        if hasattr(model, 'predict_proba'):
            all_scores = model.predict_proba(all_features)[:, 1]
        else:
            all_scores = model.predict(all_features)
        offsets = np.cumsum([0] + sizes)
        batched_scores = [all_scores[offsets[i]:offsets[i + 1]] for i in range(len(sizes))]

    for i, entry in enumerate(iterator):
        candidates = entry['candidate_ids']
        golden_ids = entry['golden_ids']

        if model is None:
            scores = entry['bm25_scores']
        elif model == 'tfidf':
            scores = entry['tfidf_scores']
        else:
            assert batched_scores is not None
            scores = batched_scores[i]

        ranked_idx = np.argsort(-scores)
        relevances = [1 if candidates[j] in golden_ids else 0 for j in ranked_idx]
        total_relevant = len(golden_ids)

        for k in k_values:
            results[f'NDCG@{k}'].append(ndcg_at_k(relevances, k))
            results[f'Recall@{k}'].append(recall_at_k(relevances, k, total_relevant))
        results['MRR'].append(reciprocal_rank(relevances))
        results['MAP'].append(average_precision(relevances))
        meta_rows.append(entry['metadata'])

    metrics = {m: np.mean(v) for m, v in results.items()}
    return metrics, pd.DataFrame(meta_rows)


def candidates_to_training_data(precomputed, feature_cols):
    """Convert precomputed candidates into X, y, groups for training rankers.

    This ensures training distribution matches evaluation (hard negatives
    from BM25/TF-IDF top-k, not random corpus chunks).
    """
    X_rows = []
    y_rows = []
    groups = []

    for entry in precomputed:
        candidates = entry['candidate_ids']
        golden_ids = entry['golden_ids']
        features = entry['features']

        labels = np.array([1 if cid in golden_ids else 0 for cid in candidates])

        X_rows.append(features)
        y_rows.append(labels)
        groups.append(len(candidates))

    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)
    groups = np.array(groups)

    return X, y, groups


def evaluate_full_retrieval(test_qa, bm25_index, tfidf_index, chunk_lookup,
                            model, feature_cols, candidate_k=100,
                            k_values=[5, 10, 20]):
    """Evaluate against full corpus (convenience wrapper, not for multi-model use)."""
    precomputed = prepare_candidates(test_qa, bm25_index, tfidf_index,
                                     chunk_lookup, feature_cols, candidate_k)
    return evaluate_from_candidates(precomputed, model, k_values)


def print_evaluation(metrics, model_name="Model"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*50}")
    print(f"  {model_name} Evaluation Results")
    print(f"{'='*50}")
    for metric, value in sorted(metrics.items()):
        print(f"  {metric:15s}: {value:.4f}")
    print(f"{'='*50}")
