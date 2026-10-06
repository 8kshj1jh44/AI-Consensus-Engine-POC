from typing import Literal

from pydantic import BaseModel, Field


RoleType = Literal[
    "layman_linguist",
    "underwriter",
    "claims_adjuster",
    "client_communications",
    "aca_expert",
]

LineOfBusinessType = Literal[
    "personal_auto",
    "homeowners",
    "umbrella",
    "commercial_auto",
    "commercial_pnc",
    "aca_health",
]


class ConsensusContext(BaseModel):
    role: RoleType
    line_of_business: LineOfBusinessType
    state: str = Field(default="MT", min_length=2)


class ConsensusRequest(BaseModel):
    prompt: str = Field(min_length=1)
    context: ConsensusContext


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0


class ProviderResult(BaseModel):
    status: Literal["success", "failed"]
    model: str
    duration_seconds: float
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    response_text: str | None = None
    error: str | None = None
    fallback_used: bool = False
    primary_error: str | None = None


class Telemetry(BaseModel):
    total_duration_seconds: float
    total_estimated_cost_usd: float
    successful_providers: list[str]
    failed_providers: list[str]


class ConsensusResponse(BaseModel):
    status: Literal["success", "failed"]
    prompt: str
    applied_context: ConsensusContext
    consensus_answer: str
    arbiter: ProviderResult
    telemetry: Telemetry
    provider_breakdown: dict[str, ProviderResult]