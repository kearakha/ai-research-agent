"""Cross-run memory: append-only JSONL log of past runs, plus keyword-overlap
recall for "have I researched something like this before". No embeddings —
ponytail: word-overlap is enough for a personal log of a few dozen runs;
swap for real semantic search (embeddings + vector store) if the log grows
past what keyword matching can tell apart.
"""
import json
import os
import re

DEFAULT_PATH = "memory.jsonl"

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_STOPWORDS = {  # small set so common connector words don't drive the match
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "is", "about",
    "yang", "dan", "atau", "di", "ke", "untuk", "dari", "ini", "itu", "aku",
    "saya", "mau", "biar", "bisa", "karena", "dengan", "tentang", "mengenai",
}


def _words(text):
    return {
        w.lower() for w in _WORD_RE.findall(text or "")
        if len(w) > 2 and w.lower() not in _STOPWORDS
    }


def load(path=DEFAULT_PATH):
    """Past runs, oldest first. Missing file -> []. A corrupt line is
    skipped, not fatal — one bad append shouldn't lose the whole log."""
    if not os.path.exists(path):
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def find_similar(goal, entries, min_overlap=0.3):
    """Best keyword-overlap match for `goal` among past `entries`, or None.
    Overlap = |shared words| / |goal words| — a short goal needs most of its
    words matched, not just one in common with a long past goal."""
    goal_words = _words(goal)
    if not goal_words:
        return None
    best, best_score = None, 0.0
    for entry in entries:
        past_words = _words(entry.get("goal", ""))
        if not past_words:
            continue
        overlap = len(goal_words & past_words) / len(goal_words)
        if overlap > best_score:
            best, best_score = entry, overlap
    return best if best_score >= min_overlap else None


def append(entry, path=DEFAULT_PATH):
    with open(path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
