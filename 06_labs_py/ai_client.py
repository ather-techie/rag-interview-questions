"""Provider-agnostic LLM client for the labs.

Reads AI_PROVIDER / AI_MODEL / AI_API_KEY (+ optional AI_BASE_URL / AI_API_VERSION)
from the environment (via .env) and dispatches to the matching SDK. Supported
AI_PROVIDER values: claude, gemini, openai, azure, ollama.
"""
import json
import os

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.environ.get("AI_PROVIDER", "gemini").lower()
MODEL = os.environ.get("AI_MODEL")
API_KEY = os.environ.get("AI_API_KEY")
BASE_URL = os.environ.get("AI_BASE_URL")
API_VERSION = os.environ.get("AI_API_VERSION", "2024-10-21")

_VALID_PROVIDERS = {"claude", "gemini", "openai", "azure", "ollama"}


def _require_known_provider():
    if PROVIDER not in _VALID_PROVIDERS:
        raise ValueError(
            f"Unknown AI_PROVIDER: {PROVIDER!r}. Expected one of: {', '.join(sorted(_VALID_PROVIDERS))}."
        )


# ---------------------------------------------------------------------------
# gemini
# ---------------------------------------------------------------------------

def _gemini_client():
    from google import genai
    return genai.Client(api_key=API_KEY)


def _gemini_to_content(msg):
    from google.genai import types

    if msg["role"] == "tool":
        return types.Content(
            role="user",
            parts=[types.Part.from_function_response(name=msg["name"], response={"result": msg["content"]})],
        )
    if msg["role"] == "assistant" and msg.get("tool_calls"):
        parts = [
            types.Part.from_function_call(name=tc["name"], args=tc["args"])
            for tc in msg["tool_calls"]
        ]
        return types.Content(role="model", parts=parts)
    role = "model" if msg["role"] == "assistant" else "user"
    return types.Content(role=role, parts=[types.Part(text=msg["content"] or "")])


def _chat_gemini(messages, tools, system, max_tokens, json_mode):
    from google.genai import types

    client = _gemini_client()
    contents = [_gemini_to_content(m) for m in messages]
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
        tools=[types.Tool(function_declarations=tools)] if tools else None,
        response_mime_type="application/json" if json_mode else None,
    )
    response = client.models.generate_content(model=MODEL, contents=contents, config=config)
    candidate = response.candidates[0].content
    function_calls = [p.function_call for p in candidate.parts if p.function_call]
    text = "".join(p.text for p in candidate.parts if p.text)
    tool_calls = [
        {"id": f"call_{i}", "name": fc.name, "args": dict(fc.args)}
        for i, fc in enumerate(function_calls)
    ]
    return {
        "text": text,
        "tool_calls": tool_calls,
        "usage": {
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
        },
    }


# ---------------------------------------------------------------------------
# claude
# ---------------------------------------------------------------------------

def _claude_client():
    import anthropic
    return anthropic.Anthropic(api_key=API_KEY)


def _claude_messages(messages):
    """Canonical messages -> Anthropic messages, merging consecutive tool results
    into a single user turn (Anthropic expects them batched, not one-per-turn)."""
    converted = []
    pending_tool_results = []

    def flush():
        if pending_tool_results:
            converted.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for msg in messages:
        if msg["role"] == "tool":
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": msg["tool_call_id"],
                "content": msg["content"],
            })
            continue
        flush()
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            content = []
            if msg.get("content"):
                content.append({"type": "text", "text": msg["content"]})
            for tc in msg["tool_calls"]:
                content.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["args"]})
            converted.append({"role": "assistant", "content": content})
        else:
            role = "assistant" if msg["role"] == "assistant" else "user"
            converted.append({"role": role, "content": msg["content"] or ""})
    flush()
    return converted


def _chat_claude(messages, tools, system, max_tokens, json_mode):
    client = _claude_client()
    if json_mode:
        system = (system or "") + "\nRespond with valid JSON only, no prose."
    kwargs = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": _claude_messages(messages),
    }
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in tools
        ]
    response = client.messages.create(**kwargs)
    text = "".join(b.text for b in response.content if b.type == "text")
    tool_calls = [
        {"id": b.id, "name": b.name, "args": b.input}
        for b in response.content if b.type == "tool_use"
    ]
    return {
        "text": text,
        "tool_calls": tool_calls,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


# ---------------------------------------------------------------------------
# openai / azure / ollama (share the OpenAI SDK shape)
# ---------------------------------------------------------------------------

def _openai_family_client():
    import openai

    if PROVIDER == "azure":
        return openai.AzureOpenAI(azure_endpoint=BASE_URL, api_version=API_VERSION, api_key=API_KEY)
    if PROVIDER == "ollama":
        return openai.OpenAI(base_url=BASE_URL or "http://localhost:11434/v1", api_key=API_KEY or "ollama")
    return openai.OpenAI(api_key=API_KEY)


def _openai_messages(messages, system, json_mode):
    converted = []
    if system:
        if json_mode:
            system = system + "\nRespond with valid JSON only, no prose."
        converted.append({"role": "system", "content": system})
    elif json_mode:
        converted.append({"role": "system", "content": "Respond with valid JSON only, no prose."})

    for msg in messages:
        if msg["role"] == "tool":
            converted.append({
                "role": "tool",
                "tool_call_id": msg["tool_call_id"],
                "content": msg["content"],
            })
        elif msg["role"] == "assistant" and msg.get("tool_calls"):
            converted.append({
                "role": "assistant",
                "content": msg.get("content"),
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])},
                    }
                    for tc in msg["tool_calls"]
                ],
            })
        else:
            converted.append({"role": msg["role"], "content": msg["content"]})
    return converted


def _chat_openai_family(messages, tools, system, max_tokens, json_mode):
    client = _openai_family_client()
    kwargs = {
        "model": MODEL,
        "messages": _openai_messages(messages, system, json_mode),
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
            for t in tools
        ]
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    message = response.choices[0].message
    tool_calls = [
        {"id": tc.id, "name": tc.function.name, "args": json.loads(tc.function.arguments)}
        for tc in (message.tool_calls or [])
    ]
    return {
        "text": message.content or "",
        "tool_calls": tool_calls,
        "usage": {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def chat(messages, tools=None, system=None, max_tokens=800, json_mode=False):
    """Multi-turn chat with optional tool/function calling.

    `messages` is a canonical list of dicts:
      {"role": "user"|"assistant"|"tool", "content": str,
       "tool_calls": [{"id","name","args"}]  (assistant turns that call tools),
       "tool_call_id": str                    (tool-role turns, id of the call being answered)}

    Returns {"text": str, "tool_calls": [{"id","name","args"}], "usage": {"input_tokens","output_tokens"}}.
    Append the returned assistant turn (text and/or tool_calls) plus one "tool"-role
    message per dispatched call back onto `messages` before the next call.
    """
    _require_known_provider()
    if PROVIDER == "gemini":
        return _chat_gemini(messages, tools, system, max_tokens, json_mode)
    if PROVIDER == "claude":
        return _chat_claude(messages, tools, system, max_tokens, json_mode)
    return _chat_openai_family(messages, tools, system, max_tokens, json_mode)


def generate(prompt, system=None, max_tokens=512, json_mode=False):
    """Single-turn text generation. Returns the reply text (or JSON string if json_mode=True)."""
    result = chat([{"role": "user", "content": prompt}], system=system, max_tokens=max_tokens, json_mode=json_mode)
    return result["text"]


def get_judge_llm():
    """A langchain chat model matching AI_PROVIDER, for use as a RAGAS judge."""
    _require_known_provider()
    if PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=MODEL, google_api_key=API_KEY)
    if PROVIDER == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=MODEL, api_key=API_KEY)
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=MODEL, api_key=API_KEY)
    if PROVIDER == "azure":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=MODEL, azure_endpoint=BASE_URL, api_version=API_VERSION, api_key=API_KEY
        )
    from langchain_ollama import ChatOllama
    return ChatOllama(model=MODEL, base_url=BASE_URL or "http://localhost:11434")
