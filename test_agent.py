"""Loop checks with a fake LLM — no network, no API key needed.

Run: python test_agent.py
"""
import json

import agent


def test_stops_on_final_action():
    replies = iter([
        '{"action": "search_web", "args": {"query": "swiss design"}}',
        '{"action": "final", "answer": "Swiss design uses grids."}',
    ])
    agent.call_llm = lambda messages: next(replies)
    agent.tools.execute_tool = lambda name, args: {"results": [{"url": "http://a"}]}

    state = agent.run("goal")

    assert state["stop_reason"] == "llm_done"
    assert state["iteration"] == 2
    assert state["answer"] == "Swiss design uses grids."
    assert state["sources"] == ["http://a"]


def test_respects_max_tool_calls():
    agent.MAX_TOOL_CALLS = 3
    agent.MAX_ITERATIONS = 99
    agent.call_llm = lambda messages: '{"action": "search_web", "args": {"query": "x"}}'
    agent.tools.execute_tool = lambda name, args: {"results": []}

    state = agent.run("goal")

    # loop stopped at the cap (never a 4th tool call); forced-final then runs
    assert state["tool_call_count"] == 3
    assert state["stop_reason"] == "forced_final"


def test_observation_fed_back_as_user_message():
    captured = {}
    replies = iter([
        '{"action": "read_page", "args": {"url": "http://p"}}',
        '{"action": "final", "answer": "done"}',
    ])
    agent.call_llm = lambda messages: (captured.setdefault("messages", messages), next(replies))[1]
    agent.tools.execute_tool = lambda name, args: {"url": "http://p", "text": "hello"}

    agent.run("goal")

    obs = [m for m in captured["messages"]
           if m["role"] == "user" and m["content"].startswith("OBSERVATION")]
    assert len(obs) == 1
    assert "hello" in obs[0]["content"]


def test_parse_action_tolerates_fence_and_prose():
    assert agent.parse_action('{"action": "final", "answer": "x"}')["action"] == "final"
    assert agent.parse_action('```json\n{"action": "search_web", "args": {}}\n```')["action"] == "search_web"
    assert agent.parse_action('Sure! {"action": "read_page", "args": {"url": "u"}} ok')["action"] == "read_page"
    assert agent.parse_action("no json here") is None
    assert agent.parse_action('{"foo": "bar"}') is None


def test_parse_action_tolerates_trailing_braces():
    a = agent.parse_action('{"action": "search_web", "args": {"query": "x"}}}')
    assert a["action"] == "search_web" and a["args"]["query"] == "x"
    assert agent.parse_action('{"action": "final", "answer": "a}b"}')["answer"] == "a}b"


def test_forced_final_when_limit_hit():
    agent.MAX_ITERATIONS = 2
    agent.MAX_TOOL_CALLS = 99
    calls = []

    def fake(messages):
        calls.append(1)
        if len(calls) <= 2:
            return '{"action": "search_web", "args": {"query": "x"}}'
        return '{"action": "final", "answer": "forced summary"}'

    agent.call_llm = fake
    agent.tools.execute_tool = lambda name, args: {"results": []}

    state = agent.run("goal")

    assert state["stop_reason"] == "forced_final"
    assert state["answer"] == "forced summary"
    assert len(calls) == 3


def test_unparseable_reply_reprompts_then_continues():
    replies = iter(["not json at all", '{"action": "final", "answer": "recovered"}'])
    agent.call_llm = lambda messages: next(replies)

    state = agent.run("goal")

    assert state["stop_reason"] == "llm_done"
    assert state["answer"] == "recovered"
    assert state["iteration"] == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
