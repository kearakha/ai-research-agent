"""Research agent loop (prompt-based tool calling).

goal (CLI arg) -> LLM picks an action -> tool runs -> observation -> repeat,
until the LLM replies with a "final" action or a hard limit is hit. The LLM
talks to us in plain JSON (one object per turn) instead of the OpenAI `tools`
API, because z-ai/glm-5.3-free on TokenRouter hangs / returns empty when the
`tools` param is present. State is a plain dict in memory. Progress is logged
as [AGENT]/[TOOL] lines to stderr. Output: report.json + report.md.
"""
import json
import os
import sys
import time

import requests

import tools

# Tunable knobs — raise if the agent stops before it has enough, lower if it
# wanders. max_iterations bounds LLM round-trips; max_tool_calls bounds total
# search_web + read_page calls across the whole run.
MAX_ITERATIONS = 10
MAX_TOOL_CALLS = 20

LLM_TIMEOUT = 120
LLM_RETRIES = 3

SYSTEM_PROMPT = """You are a research agent. You investigate the user's goal step by step.

On every turn reply with EXACTLY ONE JSON object and nothing else — no prose, no markdown fences. One of:

  {"action": "search_web", "args": {"query": "<search terms>"}}
  {"action": "read_page", "args": {"url": "<url from an earlier search result>"}}
  {"action": "final", "answer": "<thorough answer grounded in what you found, citing the URLs you used>"}

Rules:
- Start by searching, then read the most relevant pages for detail.
- After each action you get an "OBSERVATION" message. Use it to choose the next action.
- If a tool returns an error, adapt: try a different query/url, or answer from your own knowledge.
- When you have enough information, reply with "final". The answer must be detailed and cite URLs.
"""


def load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def call_llm(messages):
    base_url = os.environ.get("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    api_key = os.environ.get("TOKENROUTER_API_KEY")
    model = os.environ.get("TOKENROUTER_MODEL", "z-ai/glm-5.3-free")
    if not api_key:
        sys.exit("TOKENROUTER_API_KEY is not configured (set it in .env)")
    # ponytail: 3 tries with linear backoff — the free tier throws random
    # timeouts/503s; without this one bad call kills a whole run.
    for attempt in range(LLM_RETRIES):
        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": messages},
                timeout=LLM_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"].get("content") or ""
        except requests.RequestException as e:
            if attempt == LLM_RETRIES - 1:
                raise
            log(f"[AGENT] llm call failed ({e}), retrying")
            time.sleep(5 * (attempt + 1))


def _first_json_object(text):
    """Slice out the first brace-balanced {...} block, ignoring braces inside
    strings. Tolerates leading prose, ```json fences, and trailing junk like
    the extra "}}" the model sometimes appends."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_action(content):
    """Pull one {"action": ...} object out of the LLM reply. Returns the dict
    or None."""
    candidate = _first_json_object(content.strip())
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate, strict=False)  # gemini puts raw newlines in strings
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("action"), str):
        return obj
    return None


def _collect_sources(name, args, result, sources):
    if name == "read_page" and isinstance(args.get("url"), str):
        sources.append(args["url"])
    if name == "search_web":
        for r in result.get("results", []):
            if r.get("url"):
                sources.append(r["url"])


def run(goal):
    state = {
        "goal": goal,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"GOAL: {goal}"},
        ],
        "findings": [],
        "sources": [],
        "iteration": 0,
        "tool_call_count": 0,
        "stop_reason": None,
        "answer": "",
    }
    messages = state["messages"]

    while (
        state["iteration"] < MAX_ITERATIONS
        and state["tool_call_count"] < MAX_TOOL_CALLS
    ):
        state["iteration"] += 1
        log(f"[AGENT] iteration {state['iteration']}")
        content = call_llm(messages)
        messages.append({"role": "assistant", "content": content})
        state["findings"].append(content)

        action = parse_action(content)
        if action is None:
            log("[AGENT] unparseable reply, asking for valid JSON")
            messages.append({
                "role": "user",
                "content": "That was not valid JSON. Reply with exactly one JSON "
                           "object: search_web, read_page, or final.",
            })
            continue

        if action["action"] == "final":
            state["answer"] = str(action.get("answer", ""))
            state["stop_reason"] = "llm_done"
            break

        name = action["action"]
        args = action.get("args") or {}
        log(f"[TOOL] {name} {json.dumps(args, ensure_ascii=False)}")
        result = tools.execute_tool(name, args)
        state["tool_call_count"] += 1
        _collect_sources(name, args, result, state["sources"])
        messages.append({
            "role": "user",
            "content": f"OBSERVATION ({name}): {json.dumps(result, ensure_ascii=False)}",
        })
    else:
        state["stop_reason"] = (
            "max_tool_calls"
            if state["tool_call_count"] >= MAX_TOOL_CALLS
            else "max_iterations"
        )

    if state["stop_reason"] != "llm_done":
        log("[AGENT] limit reached without final; forcing a summary")
        messages.append({
            "role": "user",
            "content": 'You are out of steps. Reply NOW with one JSON object '
                       '{"action": "final", "answer": "..."} summarizing '
                       "everything you found, citing the URLs you read.",
        })
        # ponytail: one guard here, not on every call_llm — this extra call is
        # past the step budget and the likeliest spot for the free model to
        # flake; without it a whole run's research is lost to a crash.
        try:
            content = call_llm(messages)
        except requests.RequestException as e:
            log(f"[AGENT] forced-final call failed: {e}")
            content = ""
        state["findings"].append(content)
        action = parse_action(content)
        state["answer"] = (
            str(action.get("answer", "")) if action and action["action"] == "final"
            else content.strip()
        )
        state["stop_reason"] = "forced_final"

    state["sources"] = list(dict.fromkeys(state["sources"]))
    log(
        f"[AGENT] done: {state['stop_reason']} "
        f"({state['iteration']} iterations, {state['tool_call_count']} tool calls)"
    )
    return state


def write_reports(state):
    report = {
        "goal": state["goal"],
        "stop_reason": state["stop_reason"],
        "iterations": state["iteration"],
        "tool_calls": state["tool_call_count"],
        "sources": state["sources"],
        "answer": state["answer"],
        "findings": state["findings"],
    }
    with open("report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    lines = [
        f"# {state['goal']}",
        "",
        state["answer"] or "_(no answer produced)_",
        "",
        "## Sources",
        "",
    ]
    lines += [f"- {url}" for url in state["sources"]] or ["_(none)_"]
    lines += [
        "",
        "## Process",
        "",
        f"- stop reason: {state['stop_reason']}",
        f"- iterations: {state['iteration']}",
        f"- tool calls: {state['tool_call_count']}",
        "",
    ]
    with open("report.md", "w") as f:
        f.write("\n".join(lines))


def main():
    goal = " ".join(sys.argv[1:]).strip()
    if not goal:
        sys.exit('usage: python agent.py "your research goal"')
    load_dotenv()
    state = run(goal)
    write_reports(state)
    log("[AGENT] wrote report.json + report.md")


if __name__ == "__main__":
    main()
