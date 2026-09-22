"""memory.py checks — no network, temp files only.

Run: python test_memory.py
"""
import os
import tempfile

import memory


def test_load_missing_file_returns_empty():
    assert memory.load("/tmp/does-not-exist-ai-research-agent.jsonl") == []


def test_append_then_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "memory.jsonl")
        memory.append({"goal": "a", "answer": "b"}, path)
        memory.append({"goal": "c", "answer": "d"}, path)
        entries = memory.load(path)
        assert [e["goal"] for e in entries] == ["a", "c"]


def test_load_skips_corrupt_lines():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "memory.jsonl")
        with open(path, "w") as f:
            f.write('{"goal": "ok"}\n')
            f.write("not json\n")
            f.write('{"goal": "ok2"}\n')
        entries = memory.load(path)
        assert [e["goal"] for e in entries] == ["ok", "ok2"]


def test_find_similar_matches_overlapping_goal():
    entries = [{"goal": "riset Swiss Modernism Design buat poster instagram", "answer": "x"}]
    hit = memory.find_similar("cari info Swiss Modernism Design buat bikin poster IG", entries)
    assert hit is not None
    assert hit["goal"].startswith("riset Swiss")


def test_find_similar_no_match_for_unrelated_goal():
    entries = [{"goal": "riset Swiss Modernism Design", "answer": "x"}]
    assert memory.find_similar("cara masak rendang padang", entries) is None


def test_find_similar_empty_entries_or_goal():
    assert memory.find_similar("apa saja itu", []) is None
    assert memory.find_similar("", [{"goal": "swiss design", "answer": "x"}]) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
