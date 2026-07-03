"""
01_prepare_data.py — Baseline v2: Real Wikipedia Passage Retrieval Corpus (FAST version)

Uses pre-built word index for O(1) answer-in-passage lookup instead of O(N) linear scan.
"""

import json
import os
import random
import re
import sys
import time
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
NUM_NQ_QUESTIONS = 500
NUM_TRIVIAQA_QUESTIONS = 500
NUM_DISTRACTORS = 49
WIKI_POOL_SIZE = 20000
MIN_PASSAGE_LENGTH = 80
MAX_PASSAGE_LENGTH = 1500
RANDOM_SEED = 42
OUTPUT_DIR = "experiments/baseline_v2/data"
CACHE_DIR = "experiments/baseline_v2/cache"


def normalize(text):
    return re.sub(r'\s+', ' ', text.lower().strip())


def build_word_index(passages):
    """Build inverted index: word -> set of passage indices. O(N) build, O(1) lookup."""
    print("  Building passage word index for fast lookup...")
    index = defaultdict(set)
    for i, p in enumerate(passages):
        words = set(normalize(p).split())
        for w in words:
            if len(w) >= 3:
                index[w].add(i)
    print(f"  Index built: {len(index)} unique words across {len(passages)} passages")
    return index


def find_gold_fast(answer, passages, word_index):
    """Find passages containing answer using inverted index. Much faster than linear scan."""
    answer_norm = normalize(answer)
    answer_words = [w for w in answer_norm.split() if len(w) >= 3]
    
    if not answer_words:
        return []
    
    # Get candidate passages that contain ALL answer words
    candidates = None
    for w in answer_words:
        if w in word_index:
            if candidates is None:
                candidates = word_index[w].copy()
            else:
                candidates &= word_index[w]
        else:
            return []  # Word not in any passage
    
    if not candidates:
        return []
    
    # Verify exact substring match on candidates only
    matches = []
    for idx in candidates:
        if answer_norm in normalize(passages[idx]):
            matches.append(idx)
            if len(matches) >= 3:
                break
    return matches


def construct_gold(question, answer):
    """Natural-sounding constructed passage when no real one found."""
    templates = [
        f"{answer} is widely recognized in this context. According to various sources, "
        f"the topic relates to {answer}, which has been documented extensively in academic "
        f"and popular literature. Multiple references confirm this established fact.",
        f"Historical records and encyclopedic references indicate that {answer} plays a "
        f"significant role. This has been confirmed through multiple independent sources "
        f"and scholarly publications across different time periods.",
        f"Research and documentation confirm that {answer} is the established answer to "
        f"this topic. This fact has been verified across multiple reliable sources including "
        f"academic databases and reference materials.",
    ]
    return random.choice(templates)


def load_wiki_pool(cache_path):
    """Load cached wiki pool or download fresh."""
    if os.path.exists(cache_path):
        print(f"  Loading cached passage pool...")
        with open(cache_path, "r", encoding="utf-8") as f:
            passages = json.load(f)
        print(f"  Loaded {len(passages)} cached passages")
        return passages

    print("  Downloading Wikipedia passages (first time only)...")
    t0 = time.time()
    wiki = load_dataset("wiki_snippets", "wiki40b_en_100_0", split="train", trust_remote_code=True)
    
    random.seed(RANDOM_SEED)
    indices = random.sample(range(len(wiki)), min(WIKI_POOL_SIZE * 2, len(wiki)))
    
    passages = []
    for idx in tqdm(indices, desc="Extracting"):
        item = wiki[idx]
        text = ""
        for key in ["snippet_text", "passage_text", "section_content", "text"]:
            if key in item and isinstance(item[key], str) and len(item[key]) > 50:
                text = item[key].strip()
                break
        if not text:
            for key in item:
                if isinstance(item[key], str) and len(item[key]) > 50:
                    text = item[key].strip()
                    break
        if MIN_PASSAGE_LENGTH <= len(text) <= MAX_PASSAGE_LENGTH:
            passages.append(text)
        if len(passages) >= WIKI_POOL_SIZE:
            break

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False)
    print(f"  Saved {len(passages)} passages in {time.time()-t0:.1f}s")
    return passages


def prepare_dataset(ds_name, dataset, wiki_pool, word_index, num_questions, seed):
    """Prepare one dataset with fast gold passage lookup."""
    print(f"\n{'='*60}")
    print(f"PREPARING {ds_name} — v2 Real Passages")
    print(f"{'='*60}")

    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(num_questions, len(dataset)))
    
    prepared = []
    stats = {"real_gold": 0, "constructed": 0, "skipped": 0}

    for idx in tqdm(indices, desc=f"Preparing {ds_name}"):
        item = dataset[idx]
        question = item["question"]
        
        # Extract answers
        if ds_name == "nq":
            answers = item["answer"] if isinstance(item["answer"], list) else [item["answer"]]
        else:  # triviaqa
            answer_data = item.get("answer", {})
            if isinstance(answer_data, dict):
                aliases = answer_data.get("aliases", [])
                value = answer_data.get("value", "")
                answers = list(set([value] + aliases)) if value else aliases
            elif isinstance(answer_data, str):
                answers = [answer_data]
            else:
                answers = [str(answer_data)]
        
        answers = [a for a in answers if a and len(a.strip()) >= 2]
        if not answers:
            stats["skipped"] += 1
            continue

        # Fast gold passage lookup
        gold_idx = None
        matched_answer = None
        for ans in answers:
            matches = find_gold_fast(ans, wiki_pool, word_index)
            if matches:
                gold_idx = matches[0]
                matched_answer = ans
                break

        if gold_idx is not None:
            gold_passage = wiki_pool[gold_idx]
            gold_type = "real_wikipedia"
            stats["real_gold"] += 1
            exclude = set(matches)
        else:
            gold_passage = construct_gold(question, answers[0])
            gold_type = "constructed"
            stats["constructed"] += 1
            exclude = set()

        # Sample distractors (real Wikipedia passages)
        available = [i for i in range(len(wiki_pool)) if i not in exclude]
        distractor_indices = random.sample(available, min(NUM_DISTRACTORS, len(available)))
        distractors = [wiki_pool[i] for i in distractor_indices]

        # Insert gold at random position
        corpus = distractors.copy()
        gold_pos = random.randint(0, len(corpus))
        corpus.insert(gold_pos, gold_passage)

        prepared.append({
            "question": question,
            "answers": answers,
            "gold_passage": gold_passage,
            "gold_type": gold_type,
            "gold_position": gold_pos,
            "corpus": corpus,
            "corpus_size": len(corpus),
        })

    total = len(prepared)
    print(f"\n  Prepared: {total} questions")
    print(f"  Real Wikipedia gold: {stats['real_gold']} ({stats['real_gold']/max(1,total)*100:.1f}%)")
    print(f"  Constructed gold:    {stats['constructed']} ({stats['constructed']/max(1,total)*100:.1f}%)")
    print(f"  Skipped:             {stats['skipped']}")
    return prepared, stats


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("="*60)
    print("BASELINE v2 DATA PREPARATION — Real Wikipedia Passages (FAST)")
    print("="*60)

    # Load pool + build index
    wiki_pool = load_wiki_pool(os.path.join(CACHE_DIR, "wiki_passage_pool.json"))
    word_index = build_word_index(wiki_pool)

    # Load QA datasets
    print("\nLoading NQ-Open...")
    nq_ds = load_dataset("nq_open", split="validation", trust_remote_code=True)
    print(f"  {len(nq_ds)} questions")

    print("Loading TriviaQA...")
    tqa_ds = load_dataset("trivia_qa", "rc.nocontext", split="validation", trust_remote_code=True)
    print(f"  {len(tqa_ds)} questions")

    # Prepare
    nq_data, nq_stats = prepare_dataset("nq", nq_ds, wiki_pool, word_index, NUM_NQ_QUESTIONS, RANDOM_SEED)
    tqa_data, tqa_stats = prepare_dataset("triviaqa", tqa_ds, wiki_pool, word_index, NUM_TRIVIAQA_QUESTIONS, RANDOM_SEED + 1)

    # Save
    nq_path = os.path.join(OUTPUT_DIR, "nq_prepared.json")
    tqa_path = os.path.join(OUTPUT_DIR, "triviaqa_prepared.json")

    with open(nq_path, "w", encoding="utf-8") as f:
        json.dump(nq_data, f, indent=2, ensure_ascii=False)
    print(f"\n>>> Saved: {nq_path} ({len(nq_data)} questions)")

    with open(tqa_path, "w", encoding="utf-8") as f:
        json.dump(tqa_data, f, indent=2, ensure_ascii=False)
    print(f">>> Saved: {tqa_path} ({len(tqa_data)} questions)")

    # Sample
    if nq_data:
        s = nq_data[0]
        print(f"\nSample — Q: {s['question']}")
        print(f"  Answers: {s['answers']}")
        print(f"  Gold type: {s['gold_type']}")
        print(f"  Gold (200 chars): {s['gold_passage'][:200]}...")

    print(f"\n>>> DONE. Next: python experiments/baseline_v2/02_run_baselines.py")


if __name__ == "__main__":
    main()
