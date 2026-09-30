from app.schemas.models import ProviderResult


MODEL_PRICING = {
    "gemini-3.8-flash": {
        "input": 0.75,
        "output": 3.75,
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    "claude-haiku-4-5-20251001": {
        "input": 1.00,
        "output": 5.00,
    },
}


def calculate_model_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    pricing = MODEL_PRICING.get(model)

    if pricing is None:
        return 0.0

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


def calculate_provider_cost(result: ProviderResult) -> float:
    if result.status != "success":
        return 0.0

    return calculate_model_cost(
        model=result.model,
        input_tokens=result.tokens.input,
        output_tokens=result.tokens.output,
    )


def calculate_total_cost(
    provider_results: dict[str, ProviderResult],
) -> float:
    total = sum(
        calculate_provider_cost(result)
        for result in provider_results.values()
    )

    return round(total, 8)