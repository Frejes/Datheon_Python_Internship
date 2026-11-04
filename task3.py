
import ast
import json
import math
import operator
import os
import sys
import time
from pathlib import Path
from datetime import date
import time

import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_community.tools import DuckDuckGoSearchRun

WORKSPACE = Path(__file__).parent / "workspace"
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
WORKSPACE.mkdir(exist_ok=True)
MAX_STEPS = 8
MAX_LLM_RETRIES = 3
MAX_FILE_CHARS = 6000
RETRYABLE_HTTP_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class LLMRequestError(RuntimeError):
    """Raised when a transient LLM provider error persists after retries."""


_ddg = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Search the web in real time for current facts, news, prices, statistics, etc."""
    try:
        return _ddg.invoke(query)
    except Exception as exc:  # network / rate-limit errors should not crash the agent
        return f"Search failed: {exc}"



_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {n: getattr(math, n) for n in
          ("sqrt", "log", "log10", "sin", "cos", "tan", "exp", "floor", "ceil", "factorial")}
_FUNCS.update(abs=abs, round=round, min=min, max=max)
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS and not node.keywords):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression, e.g. '(1250 * 0.18) + sqrt(144)'.
    Supports + - * / // % **, parentheses, pi, e, sqrt, log, sin, cos, etc."""
    try:
        return str(_eval(ast.parse(expression.strip(), mode="eval").body))
    except Exception as exc:
        return f"Calculator error: {exc}"


@tool
def read_local_file(filename: str) -> str:
    """Read a text file from the local workspace folder. Pass a relative filename
    such as 'sales.csv' or 'notes.txt'. Pass '.' to list available files."""
    try:
        if filename.strip() in (".", "", "list"):
            files = sorted(p.name for p in WORKSPACE.iterdir() if p.is_file())
            return "Files: " + (", ".join(files) if files else "(workspace is empty)")
        path = (WORKSPACE / filename).resolve()
        if WORKSPACE.resolve() not in path.parents:
            return "Access denied: only files inside the workspace folder can be read."
        if not path.is_file():
            return f"File not found: {filename}"
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > MAX_FILE_CHARS:
            text = text[:MAX_FILE_CHARS] + "\n...[truncated]"
        return text
    except Exception as exc:
        return f"Read error: {exc}"


TOOLS = [web_search, calculator, read_local_file]
TOOL_MAP = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """You are an autonomous research agent with three tools: web_search, \
calculator and read_local_file.
Solve compound questions step by step. Before EVERY tool call, write one short sentence \
beginning with 'Thought:' explaining what you need and why that tool is the right one. \
Never do arithmetic in your head: use the calculator. Never guess facts that may be \
current: use web_search. When you have everything you need, reply with the final answer \
only, without calling a tool."""

SYSTEM_PROMPT += (
    f"\n\nToday's date is {date.today():%B %d, %Y}. Use the current year in searches; "
    "never assume the year from your training data. If a search already answers the "
    "question, stop and give the final answer. Use at most 3 searches."
)


class OpenRouterChatModel:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.tools = []

    def bind_tools(self, tools):
        self.tools = [convert_to_openai_tool(item) for item in tools]
        return self

    def invoke(self, messages):
        payload = {
            "model": self.model,
            "messages": [self._message_to_dict(message) for message in messages],
            "reasoning": {"enabled": True},
        }
        if self.tools:
            payload["tools"] = self.tools

        response = requests.post(
            OPENROUTER_CHAT_COMPLETIONS_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("OpenRouter response did not contain a chat message.") from exc

        tool_calls = []
        for call in message.get("tool_calls", []):
            try:
                arguments = json.loads(call["function"]["arguments"])
                tool_calls.append({
                    "name": call["function"]["name"],
                    "args": arguments,
                    "id": call["id"],
                    "type": "tool_call",
                })
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError("OpenRouter returned an invalid tool call.") from exc

        additional_kwargs = {}
        if "reasoning_details" in message:
            additional_kwargs["reasoning_details"] = message["reasoning_details"]
        return AIMessage(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            additional_kwargs=additional_kwargs,
        )

    @staticmethod
    def _message_to_dict(message):
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        if isinstance(message, AIMessage):
            result = {"role": "assistant", "content": message.content or None}
            if message.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["args"]),
                        },
                    }
                    for call in message.tool_calls
                ]
            if "reasoning_details" in message.additional_kwargs:
                result["reasoning_details"] = message.additional_kwargs["reasoning_details"]
            return result
        if isinstance(message, ToolMessage):
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
        raise TypeError(f"Unsupported message type: {type(message).__name__}")


def build_llm():
    if os.getenv("OPENROUTER_API_KEY"):
        return OpenRouterChatModel(
            api_key=os.environ["OPENROUTER_API_KEY"],
            model=os.getenv("AGENT_MODEL", "xingchen-agi/xing4.0-29b-a4b:free"),
        )
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=os.getenv("AGENT_MODEL", "claude-sonnet-5-5"), temperature=0)
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("AGENT_MODEL", "gpt-4o-mini"), temperature=0)
    sys.exit("Set OPENROUTER_API_KEY, ANTHROPIC_API_KEY or OPENAI_API_KEY first.")


def text_of(message) -> str:
    """Extract plain text from a model message (handles block-style content)."""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    return "".join(b.get("text", "") for b in content if isinstance(b, dict)).strip()


def invoke_with_retry(llm, messages):
    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            response = getattr(exc, "response", None)
            status_code = getattr(exc, "status_code", None)
            if status_code is None:
                status_code = getattr(response, "status_code", None)
            if status_code not in RETRYABLE_HTTP_STATUS_CODES:
                raise
            if attempt == MAX_LLM_RETRIES:
                raise LLMRequestError(
                    f"LLM provider returned HTTP {status_code} after "
                    f"{attempt + 1} attempts: {exc}"
                ) from exc

            delay = min(2 ** attempt, 8)
            print(
                f"[RETRY] LLM provider returned HTTP {status_code}; "
                f"retrying in {delay}s ({attempt + 1}/{MAX_LLM_RETRIES})."
            )
            time.sleep(delay)


def run_agent(llm, question: str) -> str:
    llm_with_tools = llm.bind_tools(TOOLS)
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(question)]

    for step in range(1, MAX_STEPS + 1):
        ai = invoke_with_retry(llm_with_tools, messages)
        messages.append(ai)

        if not ai.tool_calls:  # model is done
            answer = text_of(ai)
            print(f"\n[FINAL ANSWER]\n{answer}\n")
            return answer

        thought = text_of(ai)
        if thought:
            print(f"\n[Step {step}] THOUGHT: {thought.removeprefix('Thought:').strip()}")

        for call in ai.tool_calls:
            print(f"[Step {step}] ACTION: {call['name']}({call['args']})")
            tool_fn = TOOL_MAP.get(call["name"])
            result = tool_fn.invoke(call["args"]) if tool_fn else f"Unknown tool {call['name']}"
            shown = result if len(result) < 500 else result[:500] + " ...[trimmed]"
            print(f"[Step {step}] OBSERVATION: {shown}")
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    msg = "Stopped: reached the maximum number of steps without a final answer."
    print(f"\n[FINAL ANSWER]\n{msg}\n")
    return msg


def main():
    llm = build_llm()
    print("Multi-Tool Agent ready. Tools: web_search, calculator, read_local_file")
    print(f"Local files are read from: {WORKSPACE}")
    print("Type 'exit' to quit.\n")
    while True:
        try:
            question = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            try:
                run_agent(llm, question)
            except LLMRequestError as exc:
                print(f"\n[ERROR] {exc}\n")


if __name__ == "__main__":
    main()