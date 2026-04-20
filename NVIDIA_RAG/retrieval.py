"""
retrieval.py - Build BM25 and TF-IDF indexes over chunks for retrieval scoring.

BM25: sparse lexical matching (Okapi BM25 via sparse matrix ops)
TF-IDF: sparse semantic matching using sklearn's TfidfVectorizer + cosine similarity
"""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BM25Index:
    """
    Fast BM25 (Okapi) index using sklearn CountVectorizer + scipy sparse matrices.
    Replaces rank_bm25 with vectorized scoring — orders of magnitude faster.
    """

    def __init__(self, chunk_texts, chunk_ids, k1=1.5, b=0.75):
        """
        Args:
            chunk_texts: list of chunk text strings
            chunk_ids: list of chunk IDs (parallel to chunk_texts)
            k1: BM25 term frequency saturation parameter
            b: BM25 document length normalization parameter
        """
        self.chunk_ids = chunk_ids
        self.chunk_id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
        self.k1 = k1
        self.b = b

        # Build vocabulary and term-frequency matrix using sklearn
        self.vectorizer = CountVectorizer()
        tf_raw = self.vectorizer.fit_transform(chunk_texts)  # (n_docs, n_terms)

        n_docs = tf_raw.shape[0]

        # Document lengths (number of words per doc)
        doc_lens = tf_raw.sum(axis=1).A1  # dense array
        self.avgdl = doc_lens.mean()

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)  (BM25 variant)
        df = (tf_raw > 0).sum(axis=0).A1  # number of docs containing each term
        self.idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

        # Pre-compute the BM25-adjusted TF matrix:
        # adjusted_tf[d,t] = tf(t,d) * (k1+1) / (tf(t,d) + k1*(1 - b + b*|d|/avgdl))
        # This is a sparse matrix with same sparsity as tf_raw
        tf_float = tf_raw.astype(np.float64)
        len_norm = k1 * (1.0 - b + b * doc_lens / self.avgdl)  # (n_docs,)

        # For each non-zero entry: numerator = tf * (k1+1), denominator = tf + len_norm[doc]
        # We operate on the CSR data directly for speed
        self.adjusted_tf = tf_float.copy().tocsr()
        for i in range(n_docs):
            start, end = self.adjusted_tf.indptr[i], self.adjusted_tf.indptr[i + 1]
            data = self.adjusted_tf.data[start:end]
            data[:] = data * (k1 + 1.0) / (data + len_norm[i])

        print(f"BM25 index built: {n_docs} docs, {tf_raw.shape[1]} terms (sparse matrix)")

    def score_all(self, query):
        """
        Score all chunks against a query.
        Returns: numpy array of scores (shape: n_docs)
        """
        # Transform query to term vector
        q_vec = self.vectorizer.transform([query])  # (1, n_terms) sparse

        # Multiply IDF into query: q_idf[t] = q_count[t] * idf[t]
        # Since query terms usually appear once, this is just selecting IDF values
        q_idf = q_vec.multiply(self.idf)  # (1, n_terms) sparse

        # Score = q_idf @ adjusted_tf.T → (1, n_docs)
        scores = q_idf.dot(self.adjusted_tf.T).toarray().flatten()
        return scores

    def score(self, query):
        """
        Score all chunks against a query.
        Returns: dict of {chunk_id: bm25_score}
        """
        scores = self.score_all(query)
        return dict(zip(self.chunk_ids, scores.tolist()))

    def top_k(self, query, k=10):
        """Return top-k chunk IDs by BM25 score."""
        scores = self.score_all(query)
        top_idxs = np.argsort(scores)[-k:][::-1]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_idxs]


class TfidfIndex:
    """TF-IDF based retrieval index using sklearn."""

    def __init__(self, max_features=50000, ngram_range=(1, 2)):
        """
        Args:
            max_features: max vocabulary size for TF-IDF
            ngram_range: (min_n, max_n) for n-gram tokenization
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            sublinear_tf=True,  # apply log normalization to tf
        )
        self.chunk_ids = None
        self.chunk_id_to_idx = None
        self.tfidf_matrix = None

    def build_index(self, chunk_texts, chunk_ids):
        """
        Fit TF-IDF on all chunks and transform them.

        Args:
            chunk_texts: list of chunk text strings
            chunk_ids: list of chunk IDs
        """
        self.chunk_ids = chunk_ids
        self.chunk_id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
        print(f"Building TF-IDF index over {len(chunk_texts)} chunks...")
        self.tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)
        print(f"TF-IDF index built: {self.tfidf_matrix.shape[0]} docs, {self.tfidf_matrix.shape[1]} features")

    def encode_query(self, query):
        """Transform a single query into TF-IDF vector."""
        return self.vectorizer.transform([query])

    def score(self, query):
        """
        Score all chunks against a query using cosine similarity.
        Returns: dict of {chunk_id: cosine_score}
        """
        query_vec = self.encode_query(query)
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        return dict(zip(self.chunk_ids, scores.tolist()))

    def top_k(self, query, k=10):
        """Return top-k chunk IDs by TF-IDF cosine similarity."""
        scores = self.score(query)
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_chunks[:k]
