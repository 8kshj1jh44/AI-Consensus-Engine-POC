import asyncio
import time
from unittest.mock import patch

from app.core.prompts import (
    build_arbiter_system_prompt,
    build_worker_system_prompt,
)
from app.schemas.models import (
    ConsensusContext,
    ConsensusResponse,
    ProviderResult,
    Telemetry,
    TokenUsage,
)
from app.services.arbiter import query_arbiter
from app.services.claude_client import query_claude
from app.services.cost_tracker import calculate_total_cost
from app.services.gemini_client import query_gemini
from app.services.openai_client import query_openai


async def run_consensus_flow(
    prompt: str,
    context: ConsensusContext,
    simulate_anthropic_failure: bool = False,
) -> ConsensusResponse:
    worker_system_prompt = build_worker_system_prompt(
        role=context.role,
        line_of_business=context.line_of_business,
        state=context.state,
    )

    started_at = time.perf_counter()

    async def _failing_claude(*args, **kwargs):
        return ProviderResult(
            status="failed",
            model="claude-haiku-4-5-20251001",
            duration_seconds=0.15,
            tokens=TokenUsage(input=0, output=0),
            error="AuthenticationError: Simulated invalid Anthropic API key.",
        )

    claude_call = (
        _failing_claude(worker_system_prompt, prompt)
        if simulate_anthropic_failure
        else query_claude(worker_system_prompt, prompt)
    )

    results = await asyncio.gather(
        query_gemini(worker_system_prompt, prompt),
        query_openai(worker_system_prompt, prompt),
        claude_call,
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
        telemetry=telemetry,
        provider_breakdown=provider_results,
    )


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

    # Scenario 3: Failover Demo (Simulated Anthropic Outage)
    print("\n[Executing Scenario 3/3...]")
    s3_prompt = "What does comprehensive coverage protect against in severe winter weather?"
    s3_context = ConsensusContext(
        role="client_communications",
        line_of_business="personal_auto",
        state="MT",
    )
    resp3 = await run_consensus_flow(
        s3_prompt, s3_context, simulate_anthropic_failure=True
    )
    print_response("Scenario 3: Failover Demo (2/3 Survivors Synthesized)", resp3)


if __name__ == "__main__":
    asyncio.run(main())