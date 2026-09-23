import json
import os
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT_SECONDS = 180.0


class ResponsesHTTPClient:
    """Minimal Responses API client used when the optional OpenAI SDK is absent."""

    def __init__(
        self,
        api_key,
        base_url=None,
        timeout_seconds=None,
        opener=None,
    ):
        if not api_key:
            raise ValueError("An OpenAI API key is required.")

        self.api_key = api_key
        self.base_url = normalize_base_url(
            base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)
        )
        self.timeout_seconds = resolve_timeout_seconds(timeout_seconds)
        self.opener = opener or urlopen
        self.responses = ResponsesHTTPResource(self)


class ResponsesHTTPResource:
    def __init__(self, client):
        self.client = client

    def create(self, **payload):
        request = Request(
            f"{self.client.base_url}/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.client.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "sred-ai-analyzer/1.0",
            },
            method="POST",
        )

        try:
            with self.client.opener(
                request,
                timeout=self.client.timeout_seconds,
            ) as response:
                raw_body = response.read().decode("utf-8")
        except HTTPError as exc:
            raise RuntimeError(format_http_error(exc)) from exc
        except URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(
                f"Could not reach the OpenAI Responses API: {reason}"
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError("The OpenAI Responses API request timed out.") from exc

        try:
            response_payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "The OpenAI Responses API returned a non-JSON response."
            ) from exc

        if not isinstance(response_payload, dict):
            raise RuntimeError(
                "The OpenAI Responses API returned an unexpected response shape."
            )
        return ResponsesHTTPResult(response_payload)


class ResponsesHTTPResult:
    def __init__(self, payload):
        self.payload = payload
        self.output_text = extract_output_text(payload)
        usage = payload.get("usage") or {}
        self.usage = SimpleNamespace(
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
        )


def extract_output_text(payload):
    convenience_text = payload.get("output_text")
    if isinstance(convenience_text, str):
        return convenience_text

    text_parts = []
    for output_item in payload.get("output") or []:
        if not isinstance(output_item, dict):
            continue
        for content_item in output_item.get("content") or []:
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") != "output_text":
                continue
            text_value = content_item.get("text")
            if isinstance(text_value, str):
                text_parts.append(text_value)
    return "\n".join(text_parts)


def normalize_base_url(base_url):
    normalized = str(base_url).strip().rstrip("/")
    if not normalized.startswith(("https://", "http://")):
        raise ValueError("OPENAI_BASE_URL must start with http:// or https://.")
    return normalized


def resolve_timeout_seconds(timeout_seconds):
    value = (
        timeout_seconds
        if timeout_seconds is not None
        else os.environ.get("OPENAI_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    )
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("OPENAI_TIMEOUT_SECONDS must be a number.") from exc
    if resolved <= 0:
        raise ValueError("OPENAI_TIMEOUT_SECONDS must be greater than zero.")
    return resolved


def format_http_error(exc):
    message = ""
    try:
        raw_body = exc.read().decode("utf-8")
        payload = json.loads(raw_body)
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        if isinstance(error, dict):
            message = str(error.get("message", "")).strip()
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
        message = ""

    detail = f": {message}" if message else ""
    return f"OpenAI Responses API returned HTTP {exc.code}{detail}"
