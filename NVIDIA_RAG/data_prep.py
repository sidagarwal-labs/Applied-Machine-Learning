"""data_prep.py - Flatten dataset into (question, chunk, relevance) triples."""

import random
import pandas as pd
from tqdm import tqdm


def extract_all_chunks(dataset_split):
    """Extract all chunks into a flat list of dicts."""
    all_chunks = []
    for doc_idx, record in enumerate(tqdm(dataset_split, desc="Extracting chunks")):
        for chunk in record['chunks']:
            all_chunks.append({
                'chunk_id': f"{doc_idx}_{chunk['chunk_id']}",
                'chunk_text': chunk['text'],
                'doc_idx': doc_idx,
                'word_count': chunk['word_count'],
                'sentence_count': chunk['sentence_count'],
            })
    return all_chunks


def extract_qa_pairs(dataset_split):
    """Extract all QA pairs with relevant segment_ids."""
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
                'segment_ids': [f"{doc_idx}_{sid}" for sid in qa['segment_ids']],
                'doc_idx': doc_idx,
            })
    return all_qa


def create_relevance_triples(qa_pairs, all_chunks, neg_per_positive=4, seed=42):
    """Create (question, chunk, relevance) triples with positive and negative samples."""
    random.seed(seed)

    chunk_lookup = {c['chunk_id']: c for c in all_chunks}
    all_chunk_ids = list(chunk_lookup.keys())

    triples = []
    for qa in tqdm(qa_pairs, desc="Creating triples"):
        relevant_ids = set(qa['segment_ids'])

        #positive triples
        for chunk_id in relevant_ids:
            if chunk_id not in chunk_lookup:
                continue
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

        #negative triples (sample directly, resample on rare collision)
        n_neg = len(relevant_ids) * neg_per_positive
        neg_samples = set()
        while len(neg_samples) < n_neg:
            cid = random.choice(all_chunk_ids)
            if cid not in relevant_ids:
                neg_samples.add(cid)

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
