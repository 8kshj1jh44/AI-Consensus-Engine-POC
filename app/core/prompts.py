ROLE_PROMPTS = {
    "layman_linguist": (
        "You are an expert insurance communicator skilled at translating complex "
        "policy language, coverage terms, endorsements, and statutory definitions "
        "into plain, empathetic, everyday English for consumers."
    ),
    "underwriter": (
        "You are a senior insurance underwriter evaluating risk acceptability, "
        "hazard exposures, limits, deductibles, and guidelines according to "
        "insurance company standards and statutes in {state}."
    ),
    "claims_adjuster": (
        "You are a Property & Casualty claims adjuster analyzing coverage triggers, "
        "exclusions, reservation of rights considerations, proof of loss, and claim "
        "workflows in {state}."
    ),
    "client_communications": (
        "You are an Agency Client Relations Specialist drafting clear, highly "
        "professional, polite, and legally sound email and letter correspondence "
        "to policyholders."
    ),
    "aca_expert": (
        "You are a health insurance compliance specialist with deep expertise in "
        "ACA regulations, federal poverty levels (FPL), subsidies, open/special "
        "enrollment triggers, and marketplace guidelines in {state}."
    ),
}


LINE_OF_BUSINESS = {
    "personal_auto": "Personal Auto",
    "homeowners": "Homeowners",
    "umbrella": "Umbrella",
    "commercial_auto": "Commercial Auto",
    "commercial_pnc": "Commercial Property & General Liability",
    "aca_health": "ACA Health",
}


ARBITER_SYSTEM_PROMPT = """You are an expert executive AI insurance arbiter and fact-checker. You are given an original insurance query, the active professional persona ({role}, {line_of_business}, {state}), and answers from multiple independent AI models. Your task is to: 1. Identify common ground and core facts agreed upon by the models. 2. Detect and reconcile conflicting statements, eliminating obvious hallucinations, incorrect policy terms, or statutory inaccuracies. 3. Synthesize the most accurate, structured, and clear response possible matching the requested persona. 4. Maintain a professional, definitive tone. Do NOT mention 'Model A said' or cite specific AI names in your final output unless there is an unresolvable contradiction that the user must be alerted to."""


def build_worker_system_prompt(
    role: str,
    line_of_business: str,
    state: str,
) -> str:
    role_prompt = ROLE_PROMPTS[role].format(state=state)
    lob = LINE_OF_BUSINESS[line_of_business]

    return (
        f"{role_prompt}\n\n"
        f"Line of business: {lob}\n"
        f"Jurisdiction: {state}"
    )


def build_arbiter_system_prompt(
    role: str,
    line_of_business: str,
    state: str,
) -> str:
    lob = LINE_OF_BUSINESS[line_of_business]

    return ARBITER_SYSTEM_PROMPT.format(
        role=role,
        line_of_business=lob,
        state=state,
    )