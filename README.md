# AI Consensus Engine POC

A high-performance, asynchronous consensus engine designed for insurance workflows. The engine queries three LLM providers concurrently, handles isolated provider timeouts and failures gracefully, and uses an Arbiter model to synthesize a unified consensus response with comprehensive telemetry and token cost tracking.

---

## Approved Model Lineup

All model IDs are centralized via environment variables and configuration objects (`app/config.py`).

### Workers (Concurrent)
- **Google Gemini:** `gemini-3.8-flash`
- **OpenAI:** `gpt-4o-mini`
- **Anthropic Claude:** `claude-haiku-4-5-20251001`

### Arbiter (Synthesis & Failover)
- **Primary Provider:** Anthropic (`claude-sonnet-5-5`)
- **Fallback Provider:** OpenAI (`gpt-4o`)

---

## Architecture & Resilience

1. **Concurrent Worker Execution:** Workers execute in parallel via `asyncio.gather(..., return_exceptions=True)` under a strict timeout threshold (default: 12 seconds).
2. **Graceful Failover:**
   - Survives individual worker timeouts, HTTP 429/500 errors, or authentication issues.
   - The Arbiter synthesizes from whichever workers succeed (3/3, 2/3, or 1/3).
   - If all workers fail, returns HTTP 503.
   - If the primary Arbiter fails, automatically fails over to the fallback Arbiter.
3. **Telemetry & Cost Tracking:** Tracks end-to-end duration, per-provider execution time, token usage, and real-time USD cost estimation.

---

## Project Structure

```text
ai-consensus-engine/
├── app/
│   ├── main.py                 # FastAPI application and consensus endpoint
│   ├── config.py               # Pydantic settings and model configuration
│   ├── core/
│   │   └── prompts.py          # Personas, lines of business, and system prompts
│   ├── services/
│   │   ├── gemini_client.py    # Async Gemini client
│   │   ├── openai_client.py    # Async OpenAI client
│   │   ├── claude_client.py    # Async Anthropic client
│   │   ├── arbiter.py          # Primary & fallback Arbiter orchestration
│   │   └── cost_tracker.py     # Token usage and pricing calculations
│   └── schemas/
│       └── models.py           # Pydantic schemas for requests and responses
├── tests/
│   └── test_engine.py          # Automated test suite
├── .env.example                # Environment variables template
├── requirements.txt            # Project dependencies
├── run_demo.py                 # 3-scenario CLI demonstration script
└── README.md                   # Documentation