"""CLI demo: three consensus scenarios, including a live Anthropic
API-key corruption failover (Scenario 3)."""

import asyncio
import os
import time

from app.config import get_settings
from app.core.prompts import (
    build_arbiter_system_prompt,
    build_worker_system_prompt,
)
from app.schemas.models import (
    ConsensusContext,
    ConsensusResponse,
    ProviderResult,
    Telemetry,
)
from app.services.arbiter import query_arbiter
from app.services.claude_client import query_claude
from app.services.cost_tracker import calculate_total_cost
from app.services.gemini_client import query_gemini
from app.services.openai_client import query_openai


CORRUPTED_ANTHROPIC_KEY = (
    "sk-ant-api03-INTENTIONALLY-CORRUPTED-KEY-FOR-LIVE-FAILOVER-DEMO"
)


async def run_consensus_flow(
    prompt: str,
    context: ConsensusContext,
) -> ConsensusResponse:
    worker_system_prompt = build_worker_system_prompt(
        role=context.role,
        line_of_business=context.line_of_business,
        state=context.state,
    )

    started_at = time.perf_counter()

    results = await asyncio.gather(
        query_gemini(worker_system_prompt, prompt),
        query_openai(worker_system_prompt, prompt),
        query_claude(worker_system_prompt, prompt),
        return_exceptions=True,
    )

    provider_names = ["gemini", "openai", "claude"]
    provider_results: dict[str, ProviderResult] = {}

    for name, res in zip(provider_names, results):
        if isinstance(res, Exception):
            provider_results[name] = ProviderResult(
                status="failed",
                model="unknown",
                duration_seconds=0.0,
                error=f"{type(res).__name__}: {res}",
            )
        else:
            provider_results[name] = res

    successful_providers = [
        name for name, res in provider_results.items() if res.status == "success"
    ]
    failed_providers = [
        name for name, res in provider_results.items() if res.status != "success"
    ]

    if not successful_providers:
        return ConsensusResponse(
            status="failed",
            prompt=prompt,
            applied_context=context,
            consensus_answer="",
            arbiter=ProviderResult(
                status="failed",
                model="arbiter-skipped",
                duration_seconds=0.0,
                error=(
                    "All worker providers failed - consensus halted "
                    "(equivalent to the API's HTTP 503 response)."
                ),
            ),
            telemetry=Telemetry(
                total_duration_seconds=round(time.perf_counter() - started_at, 3),
                total_estimated_cost_usd=0.0,
                successful_providers=successful_providers,
                failed_providers=failed_providers,
            ),
            provider_breakdown=provider_results,
        )

    arbiter_system_prompt = build_arbiter_system_prompt(
        role=context.role,
        line_of_business=context.line_of_business,
        state=context.state,
    )

    arbiter_result = await query_arbiter(
        system_prompt=arbiter_system_prompt,
        original_prompt=prompt,
        provider_results=provider_results,
    )

    total_duration = time.perf_counter() - started_at

    all_tracked = dict(provider_results)
    all_tracked["arbiter"] = arbiter_result

    total_cost = calculate_total_cost(all_tracked)

    telemetry = Telemetry(
        total_duration_seconds=round(total_duration, 3),
        total_estimated_cost_usd=total_cost,
        successful_providers=successful_providers,
        failed_providers=failed_providers,
    )

    return ConsensusResponse(
        status="success" if arbiter_result.status == "success" else "failed",
        prompt=prompt,
        applied_context=context,
        consensus_answer=arbiter_result.response_text or "",
        arbiter=arbiter_result,
        telemetry=telemetry,
        provider_breakdown=provider_results,
    )


def resolve_primary_arbiter_model() -> str:
    settings = get_settings()

    if settings.arbiter_model_provider.lower().strip() == "openai":
        return settings.openai_arbiter_model

    return settings.anthropic_arbiter_model


def print_response(title: str, response: ConsensusResponse):
    print("=" * 70)
    print(f" {title.upper()} ")
    print("=" * 70)
    print(f"Status: {response.status}")
    print(
        f"Context: Role={response.applied_context.role} | "
        f"LOB={response.applied_context.line_of_business} | "
        f"State={response.applied_context.state}"
    )
    print(f"Prompt: {response.prompt}")
    print("\n--- CONSENSUS ANSWER ---")
    print(response.consensus_answer.strip())

    arbiter = response.arbiter

    if arbiter.status == "success":
        arbiter_role = (
            "PRIMARY"
            if arbiter.model == resolve_primary_arbiter_model()
            else "FALLBACK"
        )
        print("\n--- ARBITER ---")
        print(
            f"Model: {arbiter.model} ({arbiter_role}) | "
            f"{arbiter.duration_seconds}s | "
            f"Tokens: {arbiter.tokens.input} in / {arbiter.tokens.output} out"
        )
        if arbiter.fallback_used:
            print(f"Fallback reason: {arbiter.primary_error}")
    else:
        print("\n--- ARBITER ---")
        print(f"FAILED: {arbiter.error}")

    print("\n--- TELEMETRY ---")
    print(f"Total Duration: {response.telemetry.total_duration_seconds}s")
    print(f"Successful Providers: {response.telemetry.successful_providers}")
    print(f"Failed Providers: {response.telemetry.failed_providers}")
    print(f"Estimated Cost USD: ${response.telemetry.total_estimated_cost_usd:.6f}")
    print("\n--- PROVIDER BREAKDOWN ---")
    for provider, res in response.provider_breakdown.items():
        err_msg = f" (Error: {res.error})" if res.error else ""
        print(
            f"• {provider:8} [{res.status.upper()}] "
            f"Model: {res.model} | "
            f"{res.duration_seconds}s | "
            f"Tokens: {res.tokens.input} in / {res.tokens.output} out{err_msg}"
        )
    print("\n")


def corrupt_anthropic_key():
    """Swap in a corrupted ANTHROPIC_API_KEY for a live auth failure."""
    os.environ["ANTHROPIC_API_KEY"] = CORRUPTED_ANTHROPIC_KEY
    get_settings.cache_clear()


def restore_anthropic_key(original_env_value: str | None):
    if original_env_value is None:
        os.environ.pop("ANTHROPIC_API_KEY", None)
    else:
        os.environ["ANTHROPIC_API_KEY"] = original_env_value
    get_settings.cache_clear()


async def main():
    # Scenario 1: Layman Translation
    print("\n[Executing Scenario 1/3...]")
    s1_prompt = "Explain what 'subrogation' and 'actual cash value vs replacement cost' mean."
    s1_context = ConsensusContext(
        role="layman_linguist",
        line_of_business="personal_auto",
        state="MT",
    )
    resp1 = await run_consensus_flow(s1_prompt, s1_context)
    print_response("Scenario 1: Layman Translation", resp1)

    # Scenario 2: Underwriter / Claims Montana Coverage
    print("\n[Executing Scenario 2/3...]")
    s2_prompt = (
        "An insured backed a utility trailer into their attached garage door in Bozeman, MT. "
        "Explain coverage considerations for both the door and trailer under MT standards."
    )
    s2_context = ConsensusContext(
        role="claims_adjuster",
        line_of_business="homeowners",
        state="MT",
    )
    resp2 = await run_consensus_flow(s2_prompt, s2_context)
    print_response("Scenario 2: Underwriter / Claims Montana Coverage", resp2)

    # Scenario 3: Live Failover Demo (real corrupted ANTHROPIC_API_KEY).
    # Both the Claude worker and the Claude Arbiter hit a real 401
    # AuthenticationError, so synthesis falls back to the GPT-4o Arbiter.
    print("\n[Executing Scenario 3/3...]")
    print(
        "[Scenario 3] Corrupting ANTHROPIC_API_KEY for a LIVE auth failure "
        f"({CORRUPTED_ANTHROPIC_KEY})"
    )
    original_env_value = os.environ.get("ANTHROPIC_API_KEY")

    try:
        corrupt_anthropic_key()

        s3_prompt = "What does comprehensive coverage protect against in severe winter weather?"
        s3_context = ConsensusContext(
            role="client_communications",
            line_of_business="personal_auto",
            state="MT",
        )
        resp3 = await run_consensus_flow(s3_prompt, s3_context)
    finally:
        restore_anthropic_key(original_env_value)

    print_response(
        "Scenario 3: Live Failover (Corrupted ANTHROPIC_API_KEY, "
        "Arbiter Fallback)",
        resp3,
    )


if __name__ == "__main__":
    asyncio.run(main())
