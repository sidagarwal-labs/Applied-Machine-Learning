"""
data_prep.py - Flatten the NVIDIA dataset into (question, chunk, relevance) triples.

The raw dataset has documents, each with chunks and QA pairs.
Each QA pair has segment_ids pointing to relevant chunks.
We need to create training triples:
  - Positive: (question, relevant_chunk, 1)
  - Negative: (question, random_irrelevant_chunk, 0)
"""

import random
import pandas as pd
from tqdm import tqdm


def extract_all_chunks(dataset_split):
    """
    Extract all chunks from all documents into a flat list.
    Returns a list of dicts: {chunk_id, text, doc_idx, word_count, sentence_count}
    """
    all_chunks = []
    for doc_idx, record in enumerate(tqdm(dataset_split, desc="Extracting chunks")):
        for chunk in record['chunks']:
            all_chunks.append({
                'chunk_id': chunk['chunk_id'],
                'chunk_text': chunk['text'],
                'doc_idx': doc_idx,
                'word_count': chunk['word_count'],
                'sentence_count': chunk['sentence_count'],
            })
    return all_chunks


def extract_qa_pairs(dataset_split):
    """
    Extract all QA pairs with their relevant segment_ids.
    Returns a list of dicts with question, answer, metadata, and relevant chunk IDs.
    """
    all_qa = []
    for doc_idx, record in enumerate(tqdm(dataset_split, desc="Extracting QA pairs")):
        for qa in record['deduplicated_qa_pairs']:
            all_qa.append({
                'question': qa['question'],
                'answer': qa['answer'],
                'query_type': qa['query_type'],
                'reasoning_type': qa['reasoning_type'],
                'question_complexity': qa['question_complexity'],
                'hop_count': qa['hop_count'],
                'segment_ids': qa['segment_ids'],  # list of relevant chunk IDs
                'doc_idx': doc_idx,
            })
    return all_qa


def create_relevance_triples(qa_pairs, all_chunks, neg_per_positive=4, seed=42):
    """
    Create (question, chunk, relevance) triples.

    For each question:
      - Each chunk in segment_ids is a positive (relevance=1)
      - Sample `neg_per_positive` random chunks NOT in segment_ids as negatives (relevance=0)

    Args:
        qa_pairs: list of QA pair dicts (from extract_qa_pairs)
        all_chunks: list of chunk dicts (from extract_all_chunks)
        neg_per_positive: how many negatives per positive
        seed: random seed

    Returns:
        pd.DataFrame with columns:
            question, chunk_id, chunk_text, relevance, query_type, reasoning_type,
            question_complexity, hop_count, chunk_word_count, chunk_sentence_count
    """
    random.seed(seed)

    # Build a lookup: chunk_id -> chunk info
    chunk_lookup = {c['chunk_id']: c for c in all_chunks}
    all_chunk_ids = list(chunk_lookup.keys())

    triples = []
    for qa in tqdm(qa_pairs, desc="Creating triples"):
        relevant_ids = set(qa['segment_ids'])

        # Positive triples
        for chunk_id in relevant_ids:
            if chunk_id not in chunk_lookup:
                continue  # skip if chunk not found
            chunk = chunk_lookup[chunk_id]
            triples.append({
                'question': qa['question'],
                'chunk_id': chunk_id,
                'chunk_text': chunk['chunk_text'],
                'relevance': 1,
                'query_type': qa['query_type'],
                'reasoning_type': qa['reasoning_type'],
                'question_complexity': qa['question_complexity'],
                'hop_count': qa['hop_count'],
                'chunk_word_count': chunk['word_count'],
                'chunk_sentence_count': chunk['sentence_count'],
                'doc_idx': qa['doc_idx'],
            })

        # Negative triples: sample from chunks NOT in segment_ids
        neg_pool = [cid for cid in all_chunk_ids if cid not in relevant_ids]
        n_neg = min(len(relevant_ids) * neg_per_positive, len(neg_pool))
        neg_samples = random.sample(neg_pool, n_neg)

        for chunk_id in neg_samples:
            chunk = chunk_lookup[chunk_id]
            triples.append({
                'question': qa['question'],
                'chunk_id': chunk_id,
                'chunk_text': chunk['chunk_text'],
                'relevance': 0,
                'query_type': qa['query_type'],
                'reasoning_type': qa['reasoning_type'],
                'question_complexity': qa['question_complexity'],
                'hop_count': qa['hop_count'],
                'chunk_word_count': chunk['word_count'],
                'chunk_sentence_count': chunk['sentence_count'],
                'doc_idx': qa['doc_idx'],
            })

    return pd.DataFrame(triples)
