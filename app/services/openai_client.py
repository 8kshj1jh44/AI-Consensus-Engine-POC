import asyncio
import time

from openai import AsyncOpenAI

from app.config import get_settings
from app.schemas.models import ProviderResult, TokenUsage


async def query_openai(
    system_prompt: str,
    user_prompt: str,
) -> ProviderResult:
    settings = get_settings()

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    started_at = time.perf_counter()

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            ),
            timeout=settings.request_timeout_seconds,
        )

        duration = time.perf_counter() - started_at
        usage = response.usage

        return ProviderResult(
            status="success",
            model=settings.openai_model,
            duration_seconds=round(duration, 3),
            tokens=TokenUsage(
                input=usage.prompt_tokens if usage else 0,
                output=usage.completion_tokens if usage else 0,
            ),
            response_text=response.choices[0].message.content,
        )

    except Exception as exc:
        duration = time.perf_counter() - started_at

        return ProviderResult(
            status="failed",
            model=settings.openai_model,
            duration_seconds=round(duration, 3),
            error=f"{type(exc).__name__}: {exc}",
        )
