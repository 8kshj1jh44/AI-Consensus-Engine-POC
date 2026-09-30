import asyncio
import json
import time

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.config import get_settings
from app.schemas.models import ProviderResult, TokenUsage


PRIMARY_MODEL = "claude-sonnet-4-5-20250929"
FALLBACK_MODEL = "gpt-4o"


def build_arbiter_input(
    original_prompt: str,
    provider_results: dict[str, ProviderResult],
) -> str:
    surviving_responses = {}

    for provider, result in provider_results.items():
        if result.status == "success" and result.response_text:
            surviving_responses[provider] = result.response_text

    return (
        f"Original insurance query:\n{original_prompt}\n\n"
        "Independent provider responses:\n"
        f"{json.dumps(surviving_responses, indent=2)}"
    )


async def _query_claude_arbiter(
    system_prompt: str,
    arbiter_input: str,
) -> ProviderResult:
    settings = get_settings()

    client = AsyncAnthropic(
        api_key=settings.anthropic_api_key,
        default_headers={
            "anthropic-workspace-id": settings.anthropic_workspace_id,
        },
    )

    started_at = time.perf_counter()

    response = await asyncio.wait_for(
        client.messages.create(
            model=PRIMARY_MODEL,
            max_tokens=1500,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": arbiter_input,
                }
            ],
        ),
        timeout=settings.request_timeout_seconds,
    )

    duration = time.perf_counter() - started_at

    return ProviderResult(
        status="success",
        model=PRIMARY_MODEL,
        duration_seconds=round(duration, 3),
        tokens=TokenUsage(
            input=response.usage.input_tokens,
            output=response.usage.output_tokens,
        ),
        response_text=response.content[0].text,
    )


async def _query_openai_arbiter(
    system_prompt: str,
    arbiter_input: str,
) -> ProviderResult:
    settings = get_settings()

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    started_at = time.perf_counter()

    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=FALLBACK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": arbiter_input,
                },
            ],
        ),
        timeout=settings.request_timeout_seconds,
    )

    duration = time.perf_counter() - started_at
    usage = response.usage

    return ProviderResult(
        status="success",
        model=FALLBACK_MODEL,
        duration_seconds=round(duration, 3),
        tokens=TokenUsage(
            input=usage.prompt_tokens if usage else 0,
            output=usage.completion_tokens if usage else 0,
        ),
        response_text=response.choices[0].message.content,
    )


async def query_arbiter(
    system_prompt: str,
    original_prompt: str,
    provider_results: dict[str, ProviderResult],
) -> ProviderResult:
    arbiter_input = build_arbiter_input(
        original_prompt=original_prompt,
        provider_results=provider_results,
    )

    try:
        return await _query_claude_arbiter(
            system_prompt,
            arbiter_input,
        )

    except Exception:
        try:
            return await _query_openai_arbiter(
                system_prompt,
                arbiter_input,
            )

        except Exception as fallback_error:
            return ProviderResult(
                status="failed",
                model=FALLBACK_MODEL,
                duration_seconds=0.0,
                error=(
                    "Both primary and fallback arbiters failed. "
                    f"Fallback error: {type(fallback_error).__name__}: "
                    f"{fallback_error}"
                ),
            )