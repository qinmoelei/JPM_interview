from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from dotenv import load_dotenv
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

ENV_API_KEY = "APIYI_API_KEY"
ENV_BASE_URL = "APIYI_BASE_URL"
ENV_MODEL = "APIYI_MODEL"
ENV_TIMEOUT_SECONDS = "APIYI_TIMEOUT_SECONDS"
ENV_MAX_RETRIES = "APIYI_MAX_RETRIES"

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2

if TYPE_CHECKING:
    from openai import AsyncOpenAI


@dataclass(frozen=True)
class ApiyiConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_env(env_path: str | None) -> None:
    if env_path is None:
        env_path = str(_project_root() / ".env")
    load_dotenv(env_path, override=False)


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _get_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {name}: {value}") from exc


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid int for {name}: {value}") from exc


def load_apiyi_config(env_path: str | None = None) -> ApiyiConfig:
    _load_env(env_path)
    api_key = _get_required_env(ENV_API_KEY)
    base_url = _get_required_env(ENV_BASE_URL)
    model = _get_required_env(ENV_MODEL)
    timeout_seconds = _get_float_env(ENV_TIMEOUT_SECONDS, DEFAULT_TIMEOUT_SECONDS)
    max_retries = _get_int_env(ENV_MAX_RETRIES, DEFAULT_MAX_RETRIES)
    return ApiyiConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


def _import_openai():
    try:
        from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "openai is required for APIYi client usage. Install it with `pip install openai`."
        ) from exc
    return AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError


def create_async_client(config: ApiyiConfig, *, max_retries: int | None = None) -> "AsyncOpenAI":
    AsyncOpenAI, _, _, _ = _import_openai()
    retries = config.max_retries if max_retries is None else max_retries
    return AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout_seconds,
        max_retries=retries,
    )


def normalize_messages(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    if not messages:
        raise ValueError("messages must be a non-empty sequence")
    normalized: list[dict[str, str]] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, Mapping):
            raise TypeError(f"messages[{idx}] must be a mapping")
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError(f"messages[{idx}].role must be a non-empty string")
        if not isinstance(content, str) or not content:
            raise ValueError(f"messages[{idx}].content must be a non-empty string")
        normalized.append({"role": role, "content": content})
    return normalized


def _normalize_response(response: Any) -> Mapping[str, Any]:
    if hasattr(response, "model_dump"):
        data = response.model_dump()
    elif isinstance(response, Mapping):
        data = response
    else:
        raise TypeError("response must be a mapping or support model_dump()")
    if not isinstance(data, Mapping):
        raise TypeError("response normalization did not return a mapping")
    return data


def validate_chat_response(response: Any) -> Mapping[str, Any]:
    data = _normalize_response(response)
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response missing non-empty 'choices'")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ValueError("response choices[0] must be a mapping")
    if isinstance(first.get("message"), Mapping):
        content = first["message"].get("content")
    else:
        content = first.get("text")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("response choices[0] missing message content")
    return data


def extract_chat_content(response: Any) -> str:
    data = validate_chat_response(response)
    first = data["choices"][0]
    if isinstance(first.get("message"), Mapping):
        content = first["message"]["content"]
    else:
        content = first["text"]
    return content


async def request_chat_completion(
    messages: Sequence[Mapping[str, str]],
    *,
    client: AsyncOpenAI | None = None,
    config: ApiyiConfig | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    timeout_seconds: float | None = None,
) -> Mapping[str, Any]:
    _, APIConnectionError, APITimeoutError, RateLimitError = _import_openai()
    cfg = config or load_apiyi_config()
    normalized = normalize_messages(messages)
    if client is None:
        client = create_async_client(cfg, max_retries=0)

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max(1, cfg.max_retries + 1)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError, RateLimitError)),
        reraise=True,
    ):
        with attempt:
            response = await client.chat.completions.create(
                model=model or cfg.model,
                messages=normalized,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout_seconds or cfg.timeout_seconds,
            )
    return validate_chat_response(response)


async def ask_llm(
    question: str,
    *,
    env_path: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    timeout_seconds: float | None = None,
) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    config = load_apiyi_config(env_path)
    response = await request_chat_completion(
        [{"role": "user", "content": question}],
        config=config,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )
    return extract_chat_content(response)
