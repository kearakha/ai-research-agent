"""Tool layer for the research agent.

Two read-only tools: search_web and read_page. Each tool has an explicit
JSON schema (used for LLM function calling), argument validation, and
returns structured errors instead of raising — so the agent loop can
observe a failure and decide what to do next, instead of crashing.
"""
import os
import re
import html.parser
import requests

READ_PAGE_TIMEOUT = 20
SEARCH_TIMEOUT = 20
MAX_PAGE_CHARS = 8000

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information. Returns a list of "
                            "{title, url, snippet} results.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Fetch a URL and extract its readable text content. "
                            "Returns {url, text}.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]


class _TextExtractor(html.parser.HTMLParser):
    """Strips tags/script/style, keeps visible text. stdlib only —
    ponytail: naive extractor, swap for trafilatura/BS4 if pages come back
    too noisy to be useful."""

    def __init__(self):
        super().__init__()
        self._skip = False
        self.chunks = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            text = data.strip()
            if text:
                self.chunks.append(text)


def _extract_text(html_content: str) -> str:
    parser = _TextExtractor()
    parser.feed(html_content)
    return re.sub(r"\s+", " ", " ".join(parser.chunks)).strip()


def search_web(query: str) -> dict:
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        return {"error": "BRAVE_API_KEY is not configured"}
    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": 5},
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
            timeout=SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("description", ""),
            }
            for r in data.get("web", {}).get("results", [])
        ]
        return {"results": results}
    except requests.exceptions.Timeout:
        return {"error": f"search_web timed out after {SEARCH_TIMEOUT}s"}
    except requests.exceptions.RequestException as e:
        return {"error": f"search_web request failed: {e}"}


def read_page(url: str) -> dict:
    try:
        resp = requests.get(
            url,
            timeout=READ_PAGE_TIMEOUT,
            headers={"User-Agent": "ai-research-agent/0.1"},
        )
        resp.raise_for_status()
        text = _extract_text(resp.text)[:MAX_PAGE_CHARS]
        return {"url": url, "text": text}
    except requests.exceptions.Timeout:
        return {"error": f"read_page timed out after {READ_PAGE_TIMEOUT}s"}
    except requests.exceptions.RequestException as e:
        return {"error": f"read_page request failed: {e}"}


TOOL_REGISTRY = {
    "search_web": search_web,
    "read_page": read_page,
}


def validate_args(tool_name: str, args: dict) -> str | None:
    """Returns an error string if args are invalid, else None."""
    schema = next(
        (t["function"] for t in TOOL_SCHEMAS if t["function"]["name"] == tool_name),
        None,
    )
    if schema is None:
        return f"unknown tool: {tool_name}"
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    for field in schema["parameters"]["required"]:
        if field not in args:
            return f"missing required argument: {field}"
        if not isinstance(args[field], str) or not args[field].strip():
            return f"argument '{field}' must be a non-empty string"
    return None


def execute_tool(tool_name: str, args: dict) -> dict:
    """Validate then execute. Never raises — always returns a dict the
    agent can feed back to the LLM as an observation."""
    error = validate_args(tool_name, args)
    if error:
        return {"error": error}
    return TOOL_REGISTRY[tool_name](**args)
