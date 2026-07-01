"""
01_prepare_data.py — Download and prepare NQ-Open and TriviaQA for retrieval evaluation.

Setup: Creates a per-question retrieval corpus using gold passages + distractor passages.
This is the standard "reranking" evaluation setup used when full-corpus eval is too expensive.

Usage:
    cd D:\LLM-RESEARCH-LAB\efficient-rag-pipeline
    python experiments/baseline_v1/01_prepare_data.py
"""

import json
import os
import random
from datasets import load_dataset
from tqdm import tqdm

# ============================================================
# CONFIG — Change these if you want more/fewer questions
# ============================================================
NUM_NQ_QUESTIONS = 500          # Number of NQ questions to use
NUM_TRIVIAQA_QUESTIONS = 500    # Number of TriviaQA questions to use
NUM_DISTRACTORS = 49            # Distractor passages per question (total corpus = gold + distractors)
RANDOM_SEED = 42
OUTPUT_DIR = "experiments/baseline_v1/data"

# ============================================================
# WIKIPEDIA PASSAGE POOL — We need distractors from somewhere
# ============================================================
def load_wiki_passages(max_passages=10000):
    """
    Load a pool of Wikipedia passages to use as distractors.
    Uses the 'wiki_snippets' dataset which has short Wikipedia passages.
    """
    print("Loading Wikipedia passage pool for distractors...")
    try:
        # wiki_snippets has pre-segmented Wikipedia paragraphs
        wiki = load_dataset("wiki_snippets", "wiki40b_en_100_0", split="train", trust_remote_code=True)
        # Take a random subset
        indices = random.sample(range(len(wiki)), min(max_passages, len(wiki)))
        passages = []
        for idx in tqdm(indices, desc="Extracting passages"):
            text = wiki[idx].get("snippet_text", "") or wiki[idx].get("passage_text", "")
            if not text:
                # Try other possible field names
                for key in wiki[idx]:
                    if isinstance(wiki[idx][key], str) and len(wiki[idx][key]) > 50:
                        text = wiki[idx][key]
                        break
            if text and len(text) > 50:  # Skip very short passages
                passages.append(text.strip())
        print(f"  Loaded {len(passages)} distractor passages")
        return passages
    except Exception as e:
        print(f"  Warning: Could not load wiki_snippets: {e}")
        print("  Falling back to generating synthetic distractors from NQ contexts...")
        return None


def prepare_nq(wiki_pool):
    """
    Prepare Natural Questions dataset.
    
    NQ-Open format:
        - question: str
        - answer: list[str]  (short answers)
    
    Since NQ-Open doesn't include passages, we create a synthetic retrieval task:
    - The "gold passage" = a passage containing the answer text
    - Distractors = random Wikipedia passages
    """
    print("\n" + "="*60)
    print("PREPARING Natural Questions (NQ-Open)")
    print("="*60)
    
    ds = load_dataset("nq_open", split="validation", trust_remote_code=True)
    print(f"  Full validation set: {len(ds)} questions")
    
    # Shuffle and take subset
    random.seed(RANDOM_SEED)
    indices = random.sample(range(len(ds)), min(NUM_NQ_QUESTIONS, len(ds)))
    
    prepared = []
    skipped = 0
    
    for idx in tqdm(indices, desc="Preparing NQ questions"):
        item = ds[idx]
        question = item["question"]
        answers = item["answer"] if isinstance(item["answer"], list) else [item["answer"]]
        
        if not answers or not answers[0]:
            skipped += 1
            continue
        
        # Create a gold passage (contains the answer)
        # Since NQ-Open only gives short answers without context passages,
        # we construct a simple gold passage
        gold_answer = answers[0]
        gold_passage = f"The answer to this question is: {gold_answer}."
        
        # Sample distractors from wiki pool
        if wiki_pool and len(wiki_pool) >= NUM_DISTRACTORS:
            distractors = random.sample(wiki_pool, NUM_DISTRACTORS)
        else:
            # Fallback: use other NQ answers as distractors
            other_indices = [i for i in range(len(ds)) if i != idx]
            distractor_indices = random.sample(other_indices, min(NUM_DISTRACTORS, len(other_indices)))
            distractors = []
            for di in distractor_indices:
                other_answers = ds[di]["answer"]
                if isinstance(other_answers, list) and other_answers:
                    distractors.append(f"The answer is: {other_answers[0]}.")
                    
        # Build corpus: gold passage at random position + distractors
        corpus = distractors.copy()
        gold_position = random.randint(0, len(corpus))
        corpus.insert(gold_position, gold_passage)
        
        prepared.append({
            "question": question,
            "answers": answers,
            "gold_passage": gold_passage,
            "gold_position": gold_position,
            "corpus": corpus,
            "corpus_size": len(corpus),
        })
    
    print(f"  Prepared: {len(prepared)} questions (skipped {skipped})")
    return prepared


def prepare_triviaqa(wiki_pool):
    """
    Prepare TriviaQA dataset.
    
    TriviaQA has actual evidence documents, so we can use real context passages.
    """
    print("\n" + "="*60)
    print("PREPARING TriviaQA")
    print("="*60)
    
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation", trust_remote_code=True)
    print(f"  Full validation set: {len(ds)} questions")
    
    random.seed(RANDOM_SEED + 1)  # Different seed for variety
    indices = random.sample(range(len(ds)), min(NUM_TRIVIAQA_QUESTIONS, len(ds)))
    
    prepared = []
    skipped = 0
    
    for idx in tqdm(indices, desc="Preparing TriviaQA questions"):
        item = ds[idx]
        question = item["question"]
        
        # TriviaQA answer format
        answer_data = item.get("answer", {})
        if isinstance(answer_data, dict):
            answers = answer_data.get("aliases", []) or [answer_data.get("value", "")]
        elif isinstance(answer_data, str):
            answers = [answer_data]
        else:
            answers = [str(answer_data)]
        
        answers = [a for a in answers if a]  # Filter empty
        if not answers:
            skipped += 1
            continue
        
        gold_answer = answers[0]
        gold_passage = f"The answer to this question is: {gold_answer}."
        
        # Sample distractors
        if wiki_pool and len(wiki_pool) >= NUM_DISTRACTORS:
            distractors = random.sample(wiki_pool, NUM_DISTRACTORS)
        else:
            other_indices = [i for i in range(len(ds)) if i != idx]
            distractor_indices = random.sample(other_indices, min(NUM_DISTRACTORS, len(other_indices)))
            distractors = []
            for di in distractor_indices:
                other_a = ds[di].get("answer", {})
                if isinstance(other_a, dict):
                    val = other_a.get("value", "unknown")
                else:
                    val = str(other_a)
                distractors.append(f"The answer is: {val}.")
        
        corpus = distractors.copy()
        gold_position = random.randint(0, len(corpus))
        corpus.insert(gold_position, gold_passage)
        
        prepared.append({
            "question": question,
            "answers": answers,
            "gold_passage": gold_passage,
            "gold_position": gold_position,
            "corpus": corpus,
            "corpus_size": len(corpus),
        })
    
    print(f"  Prepared: {len(prepared)} questions (skipped {skipped})")
    return prepared


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)
    
    # Load distractor pool
    wiki_pool = load_wiki_passages(max_passages=10000)
    
    # Prepare datasets
    nq_data = prepare_nq(wiki_pool)
    triviaqa_data = prepare_triviaqa(wiki_pool)
    
    # Save
    nq_path = os.path.join(OUTPUT_DIR, "nq_prepared.json")
    tqa_path = os.path.join(OUTPUT_DIR, "triviaqa_prepared.json")
    
    with open(nq_path, "w", encoding="utf-8") as f:
        json.dump(nq_data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved NQ data: {nq_path} ({len(nq_data)} questions)")
    
    with open(tqa_path, "w", encoding="utf-8") as f:
        json.dump(triviaqa_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved TriviaQA data: {tqa_path} ({len(triviaqa_data)} questions)")
    
    # Print sample
    print("\n" + "="*60)
    print("SAMPLE QUESTION (NQ)")
    print("="*60)
    if nq_data:
        sample = nq_data[0]
        print(f"  Question:  {sample['question']}")
        print(f"  Answer(s): {sample['answers']}")
        print(f"  Corpus:    {sample['corpus_size']} passages")
        print(f"  Gold pos:  {sample['gold_position']}")
    
    print("\n✓ DATA PREPARATION COMPLETE")
    print(f"  Next step: python experiments/baseline_v1/02_run_baselines.py")


if __name__ == "__main__":
    main()