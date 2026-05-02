"""Faithfulness metric: fraction of answer claims supported by contexts."""
from pydantic import BaseModel, Field
from ..types import EvalSample, MetricScore
from ..llm import call_json, DEFAULT_MODEL


EXTRACT_CLAIMS_PROMPT = """Extract atomic factual claims from the answer.
Each claim should be a single, self-contained, verifiable statement.

Question: {question}
Answer: {answer}

Return ONLY valid JSON in this exact format:
{{"claims": ["claim 1", "claim 2", "..."]}}
"""

VERIFY_CLAIM_PROMPT = """Decide if a claim is supported by the given contexts.

A claim is "supported" ONLY if the contexts directly state it or clearly imply it.
A claim is "unsupported" if the contexts do not state or imply it, even if the
claim is generally true in the world.

Claim: {claim}

Contexts:
{contexts}

Return ONLY valid JSON in this exact format:
{{"supported": true, "reason": "brief explanation"}}
"""


class _ClaimsResponse(BaseModel):
    claims: list[str] = Field(default_factory=list)


class _VerifyResponse(BaseModel):
    supported: bool
    reason: str = ""


def faithfulness(sample: EvalSample, model: str = DEFAULT_MODEL) -> MetricScore:
    """Faithfulness: fraction of answer claims supported by retrieved contexts.

    Score in [0.0, 1.0]. Higher means the answer is more grounded in the
    retrieved contexts (less hallucination).

    Args:
        sample: An EvalSample with question, answer, and contexts.
        model: LLM model name in litellm format (e.g. "gpt-4o-mini",
            "groq/llama-3.3-70b-versatile").

    Returns:
        MetricScore with the faithfulness score and per-claim details.
    """
    if not sample.contexts:
        return MetricScore(
            name="faithfulness",
            score=0.0,
            details={"error": "no contexts provided"},
        )

    # Step 1: extract atomic claims from the answer.
    claims_resp = call_json(
        EXTRACT_CLAIMS_PROMPT.format(
            question=sample.question, answer=sample.answer
        ),
        _ClaimsResponse,
        model=model,
    )
    claims = [c.strip() for c in claims_resp.claims if c.strip()]

    if not claims:
        return MetricScore(
            name="faithfulness",
            score=0.0,
            details={"error": "no claims extracted from answer"},
        )

    # Step 2: verify each claim against the contexts.
    contexts_text = "\n\n".join(
        f"[Context {i + 1}]\n{c}" for i, c in enumerate(sample.contexts)
    )

    verifications = []
    supported = 0
    for claim in claims:
        v = call_json(
            VERIFY_CLAIM_PROMPT.format(claim=claim, contexts=contexts_text),
            _VerifyResponse,
            model=model,
        )
        verifications.append(
            {"claim": claim, "supported": v.supported, "reason": v.reason}
        )
        if v.supported:
            supported += 1

    score = supported / len(claims)
    return MetricScore(
        name="faithfulness",
        score=score,
        details={
            "claims_total": len(claims),
            "claims_supported": supported,
            "verifications": verifications,
        },
    )
