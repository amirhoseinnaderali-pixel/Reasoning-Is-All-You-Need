import json
from typing import List

import requests


def _collect_streaming_response(raw_text: str) -> List[str]:
    """Parse a newline-delimited JSON response body from Ollama."""
    parts: List[str] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            if "response" in payload and isinstance(payload["response"], str):
                parts.append(payload["response"])
            else:
                message = payload.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        parts.append(content)
    return parts


def chat_completion(model: str, prompt: str, api_key: str) -> str:
    # Add timeout to prevent hanging (30 seconds for connect, 120 seconds for read)
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt},
        timeout=(30, 120)  # (connect timeout, read timeout)
    )
    resp.raise_for_status()

    body = resp.text
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        parts = _collect_streaming_response(body)
        if parts:
            return "".join(parts)
        return body

    message = payload.get("response")
    if isinstance(message, str):
        return message

    nested = payload.get("message")
    if isinstance(nested, dict):
        content = nested.get("content")
        if isinstance(content, str):
            return content

    return json.dumps(payload)



