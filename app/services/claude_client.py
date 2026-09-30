import asyncio
import time

from anthropic import AsyncAnthropic

from app.config import get_settings
from app.schemas.models import ProviderResult, TokenUsage


MODEL = "claude-haiku-4-5-20251001"


async def query_claude(
    system_prompt: str,
    user_prompt: str,
) -> ProviderResult:
    settings = get_settings()

    client = AsyncAnthropic(
        api_key=settings.anthropic_api_key,
        default_headers={
            "anthropic-workspace-id": settings.anthropic_workspace_id,
        },
    )

    started_at = time.perf_counter()

    try:
        response = await asyncio.wait_for(
            client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
            ),
            timeout=settings.request_timeout_seconds,
        )

        duration = time.perf_counter() - started_at

        return ProviderResult(
            status="success",
            model=MODEL,
            duration_seconds=round(duration, 3),
            tokens=TokenUsage(
                input=response.usage.input_tokens,
                output=response.usage.output_tokens,
            ),
            response_text=response.content[0].text,
        )

    except Exception as exc:
        duration = time.perf_counter() - started_at

        return ProviderResult(
            status="failed",
            model=MODEL,
            duration_seconds=round(duration, 3),
            error=f"{type(exc).__name__}: {exc}",
        )