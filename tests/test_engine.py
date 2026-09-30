import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.schemas.models import ProviderResult, TokenUsage


client = TestClient(main_module.app)


@pytest.mark.asyncio
async def test_consensus_endpoint_survives_one_provider_failure(monkeypatch):
    async def mock_gemini(system_prompt, user_prompt):
        return ProviderResult(
            status="success",
            model="gemini-test",
            duration_seconds=1.2,
            tokens=TokenUsage(input=50, output=100),
            response_text="Gemini coverage analysis",
        )

    async def mock_openai(system_prompt, user_prompt):
        return ProviderResult(
            status="success",
            model="openai-test",
            duration_seconds=1.0,
            tokens=TokenUsage(input=50, output=100),
            response_text="OpenAI coverage analysis",
        )

    async def mock_claude(system_prompt, user_prompt):
        return ProviderResult(
            status="failed",
            model="claude-test",
            duration_seconds=0.5,
            error="Simulated Anthropic failure",
        )

    async def mock_arbiter(
        system_prompt,
        original_prompt,
        provider_results,
    ):
        assert provider_results["gemini"].status == "success"
        assert provider_results["openai"].status == "success"
        assert provider_results["claude"].status == "failed"

        return ProviderResult(
            status="success",
            model="arbiter-test",
            duration_seconds=0.8,
            tokens=TokenUsage(input=100, output=150),
            response_text="Synthesized consensus answer",
        )

    monkeypatch.setattr(
        main_module,
        "query_gemini",
        mock_gemini,
    )
    monkeypatch.setattr(
        main_module,
        "query_openai",
        mock_openai,
    )
    monkeypatch.setattr(
        main_module,
        "query_claude",
        mock_claude,
    )
    monkeypatch.setattr(
        main_module,
        "query_arbiter",
        mock_arbiter,
    )

    response = client.post(
        "/api/v1/consensus",
        json={
            "prompt": "Test insurance coverage question",
            "context": {
                "role": "claims_adjuster",
                "line_of_business": "homeowners",
                "state": "MT",
            },
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "success"
    assert body["consensus_answer"] == "Synthesized consensus answer"

    assert set(body["telemetry"]["successful_providers"]) == {
        "gemini",
        "openai",
    }

    assert body["telemetry"]["failed_providers"] == [
        "claude",
    ]

    assert body["provider_breakdown"]["claude"]["status"] == "failed"
@pytest.mark.asyncio
async def test_consensus_endpoint_returns_503_when_all_providers_fail(
    monkeypatch,
):
    async def mock_failed_provider(system_prompt, user_prompt):
        return ProviderResult(
            status="failed",
            model="test-model",
            duration_seconds=0.5,
            error="Simulated provider failure",
        )

    monkeypatch.setattr(
        main_module,
        "query_gemini",
        mock_failed_provider,
    )
    monkeypatch.setattr(
        main_module,
        "query_openai",
        mock_failed_provider,
    )
    monkeypatch.setattr(
        main_module,
        "query_claude",
        mock_failed_provider,
    )

    response = client.post(
        "/api/v1/consensus",
        json={
            "prompt": "Test insurance coverage question",
            "context": {
                "role": "claims_adjuster",
                "line_of_business": "homeowners",
                "state": "MT",
            },
        },
    )
from app.services.arbiter import build_arbiter_input
from app.services.cost_tracker import calculate_total_cost


def make_success_result(model: str, text: str) -> ProviderResult:
    return ProviderResult(
        status="success",
        model=model,
        duration_seconds=1.0,
        tokens=TokenUsage(input=100, output=200),
        response_text=text,
    )


def make_failed_result(model: str) -> ProviderResult:
    return ProviderResult(
        status="failed",
        model=model,
        duration_seconds=1.0,
        error="Simulated provider failure",
    )


def test_arbiter_receives_only_successful_providers():
    results = {
        "gemini": make_success_result(
            "gemini-3.8-flash",
            "Gemini answer",
        ),
        "openai": make_success_result(
            "gpt-4o-mini",
            "OpenAI answer",
        ),
        "claude": make_failed_result(
            "claude-haiku-4-5-20251001",
        ),
    }

    arbiter_input = build_arbiter_input(
        original_prompt="Test insurance question",
        provider_results=results,
    )

    assert "Gemini answer" in arbiter_input
    assert "OpenAI answer" in arbiter_input
    assert "Simulated provider failure" not in arbiter_input


def test_failed_provider_costs_zero():
    results = {
        "gemini": make_success_result(
            "gemini-3.8-flash",
            "Successful answer",
        ),
        "claude": make_failed_result(
            "claude-haiku-4-5-20251001",
        ),
    }

    total_cost = calculate_total_cost(results)

    assert total_cost > 0


@pytest.mark.parametrize("successful_count", [1, 2, 3])
def test_surviving_provider_count(successful_count):
    providers = ["gemini", "openai", "claude"]

    results = {
        provider: (
            make_success_result("test-model", "answer")
            if index < successful_count
            else make_failed_result("test-model")
        )
        for index, provider in enumerate(providers)
    }

    successful = [
        provider
        for provider, result in results.items()
        if result.status == "success"
    ]

    assert len(successful) == successful_count
import app.services.arbiter as arbiter_module


@pytest.mark.asyncio
async def test_arbiter_falls_back_to_openai_when_claude_fails(
    monkeypatch,
):
    async def mock_claude_arbiter(system_prompt, arbiter_input):
        raise RuntimeError("Simulated Claude Arbiter failure")

    async def mock_openai_arbiter(system_prompt, arbiter_input):
        return ProviderResult(
            status="success",
            model="gpt-4o",
            duration_seconds=0.7,
            tokens=TokenUsage(input=100, output=150),
            response_text="OpenAI fallback consensus",
        )

    monkeypatch.setattr(
        arbiter_module,
        "_query_claude_arbiter",
        mock_claude_arbiter,
    )
    monkeypatch.setattr(
        arbiter_module,
        "_query_openai_arbiter",
        mock_openai_arbiter,
    )

    provider_results = {
        "gemini": make_success_result(
            "gemini-test",
            "Gemini answer",
        ),
        "openai": make_success_result(
            "openai-test",
            "OpenAI worker answer",
        ),
        "claude": make_success_result(
            "claude-test",
            "Claude worker answer",
        ),
    }

    result = await arbiter_module.query_arbiter(
        system_prompt="Test Arbiter system prompt",
        original_prompt="Test insurance question",
        provider_results=provider_results,
    )

    assert result.status == "success"
    assert result.model == "gpt-4o"
    assert result.response_text == "OpenAI fallback consensus"
@pytest.mark.asyncio
async def test_arbiter_returns_failed_when_primary_and_fallback_fail(
    monkeypatch,
):
    async def mock_claude_arbiter(system_prompt, arbiter_input):
        raise RuntimeError("Simulated Claude Arbiter failure")

    async def mock_openai_arbiter(system_prompt, arbiter_input):
        raise RuntimeError("Simulated OpenAI Arbiter failure")

    monkeypatch.setattr(
        arbiter_module,
        "_query_claude_arbiter",
        mock_claude_arbiter,
    )
    monkeypatch.setattr(
        arbiter_module,
        "_query_openai_arbiter",
        mock_openai_arbiter,
    )

    provider_results = {
        "gemini": make_success_result(
            "gemini-test",
            "Gemini answer",
        ),
    }

    result = await arbiter_module.query_arbiter(
        system_prompt="Test Arbiter system prompt",
        original_prompt="Test insurance question",
        provider_results=provider_results,
    )

    assert result.status == "failed"
    assert result.model == "arbiter-unavailable"
    assert "Both primary and fallback arbiters failed" in result.error
@pytest.mark.asyncio
async def test_consensus_endpoint_returns_502_when_arbiter_fails(
    monkeypatch,
):
    async def mock_successful_worker(system_prompt, user_prompt):
        return ProviderResult(
            status="success",
            model="worker-test",
            duration_seconds=0.5,
            tokens=TokenUsage(input=50, output=100),
            response_text="Worker insurance analysis",
        )

    async def mock_failed_arbiter(
        system_prompt,
        original_prompt,
        provider_results,
    ):
        return ProviderResult(
            status="failed",
            model="gpt-4o",
            duration_seconds=0.5,
            error="Both primary and fallback arbiters failed.",
        )

    monkeypatch.setattr(
        main_module,
        "query_gemini",
        mock_successful_worker,
    )
    monkeypatch.setattr(
        main_module,
        "query_openai",
        mock_successful_worker,
    )
    monkeypatch.setattr(
        main_module,
        "query_claude",
        mock_successful_worker,
    )
    monkeypatch.setattr(
        main_module,
        "query_arbiter",
        mock_failed_arbiter,
    )

    response = client.post(
        "/api/v1/consensus",
        json={
            "prompt": "Test insurance coverage question",
            "context": {
                "role": "underwriter",
                "line_of_business": "personal_auto",
                "state": "MT",
            },
        },
    )

    assert response.status_code == 502