import asyncio
import time

from fastapi import FastAPI, HTTPException

from app.core.prompts import (
    build_arbiter_system_prompt,
    build_worker_system_prompt,
)
from app.schemas.models import (
    ConsensusRequest,
    ConsensusResponse,
    ProviderResult,
    Telemetry,
)
from app.services.arbiter import query_arbiter
from app.services.claude_client import query_claude
from app.services.cost_tracker import (
    calculate_model_cost,
    calculate_total_cost,
)
from app.services.gemini_client import query_gemini
from app.services.openai_client import query_openai


app = FastAPI(
    title="Multi-LLM Consensus Engine for Insurance",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post(
    "/api/v1/consensus",
    response_model=ConsensusResponse,
)
async def create_consensus(
    request: ConsensusRequest,
) -> ConsensusResponse:
    pipeline_started = time.perf_counter()

    worker_system_prompt = build_worker_system_prompt(
        role=request.context.role,
        line_of_business=request.context.line_of_business,
        state=request.context.state,
    )

    worker_tasks = [
        query_gemini(worker_system_prompt, request.prompt),
        query_openai(worker_system_prompt, request.prompt),
        query_claude(worker_system_prompt, request.prompt),
    ]

    raw_results = await asyncio.gather(
        *worker_tasks,
        return_exceptions=True,
    )

    provider_names = ["gemini", "openai", "claude"]
    provider_results: dict[str, ProviderResult] = {}

    for provider, result in zip(provider_names, raw_results):
        if isinstance(result, Exception):
            provider_results[provider] = ProviderResult(
                status="failed",
                model="unknown",
                duration_seconds=0.0,
                error=f"{type(result).__name__}: {result}",
            )
        else:
            provider_results[provider] = result

    successful_providers = [
        provider
        for provider, result in provider_results.items()
        if result.status == "success"
    ]

    failed_providers = [
        provider
        for provider, result in provider_results.items()
        if result.status == "failed"
    ]

    if not successful_providers:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "failed",
                "message": "All worker providers failed.",
                "failed_providers": failed_providers,
            },
        )

    arbiter_system_prompt = build_arbiter_system_prompt(
        role=request.context.role,
        line_of_business=request.context.line_of_business,
        state=request.context.state,
    )

    arbiter_result = await query_arbiter(
        system_prompt=arbiter_system_prompt,
        original_prompt=request.prompt,
        provider_results=provider_results,
    )

    if arbiter_result.status == "failed":
        raise HTTPException(
            status_code=502,
            detail={
                "status": "failed",
                "message": "Consensus Arbiter failed.",
                "error": arbiter_result.error,
            },
        )

    worker_cost = calculate_total_cost(provider_results)

    arbiter_cost = calculate_model_cost(
        model=arbiter_result.model,
        input_tokens=arbiter_result.tokens.input,
        output_tokens=arbiter_result.tokens.output,
    )

    total_cost = round(worker_cost + arbiter_cost, 8)
    total_duration = time.perf_counter() - pipeline_started

    return ConsensusResponse(
        status="success",
        prompt=request.prompt,
        applied_context=request.context,
        consensus_answer=arbiter_result.response_text or "",
        telemetry=Telemetry(
            total_duration_seconds=round(total_duration, 3),
            total_estimated_cost_usd=total_cost,
            successful_providers=successful_providers,
            failed_providers=failed_providers,
        ),
        provider_breakdown=provider_results,
    )