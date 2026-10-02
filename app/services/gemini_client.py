import asyncio
import time

from google import genai

from app.config import get_settings
from app.schemas.models import ProviderResult, TokenUsage



async def query_gemini(
    system_prompt: str,
    user_prompt: str,
) -> ProviderResult:
    settings = get_settings()

    client = genai.Client(api_key=settings.gemini_api_key)
    started_at = time.perf_counter()

    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=settings.gemini_model,
                contents=f"{system_prompt}\n\nUser query:\n{user_prompt}",
            ),
            timeout=settings.request_timeout_seconds,
        )

        duration = time.perf_counter() - started_at
        usage = response.usage_metadata

        return ProviderResult(
            status="success",
            model=settings.gemini_model,
            duration_seconds=round(duration, 3),
            tokens=TokenUsage(
                input=(
                    usage.prompt_token_count
                    if usage and usage.prompt_token_count
                    else 0
                ),
                output=(
                    usage.candidates_token_count
                    if usage and usage.candidates_token_count
                    else 0
                ),
            ),
            response_text=response.text,
        )

    except Exception as exc:
        duration = time.perf_counter() - started_at

        return ProviderResult(
            status="failed",
            model=settings.gemini_model,
            duration_seconds=round(duration, 3),
            error=f"{type(exc).__name__}: {exc}",
        )