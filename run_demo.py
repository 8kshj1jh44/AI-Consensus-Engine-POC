import asyncio
import time

from app.core.prompts import build_worker_system_prompt
from app.services.claude_client import query_claude
from app.services.gemini_client import query_gemini
from app.services.openai_client import query_openai


async def main():
    system_prompt = build_worker_system_prompt(
        role="claims_adjuster",
        line_of_business="homeowners",
        state="MT",
    )

    user_prompt = (
        "An insured backed a trailer into their garage door and damaged "
        "both the door and the trailer. Briefly explain the coverage considerations."
    )

    started_at = time.perf_counter()

    results = await asyncio.gather(
        query_gemini(system_prompt, user_prompt),
        query_openai(system_prompt, user_prompt),
        query_claude(system_prompt, user_prompt),
        return_exceptions=True,
    )

    total_duration = time.perf_counter() - started_at

    providers = ["gemini", "openai", "claude"]

    print("\n=== PARALLEL WORKER TEST ===")

    for provider, result in zip(providers, results):
        print(f"\n{provider.upper()}")

        if isinstance(result, Exception):
            print(f"Unhandled exception: {result}")
            continue

        print(f"Status: {result.status}")
        print(f"Model: {result.model}")
        print(f"Duration: {result.duration_seconds}s")
        print(
            f"Tokens: input={result.tokens.input}, "
            f"output={result.tokens.output}"
        )

        if result.error:
            print(f"Error: {result.error}")

    print(f"\nTotal worker phase: {total_duration:.3f}s")


if __name__ == "__main__":
    asyncio.run(main())