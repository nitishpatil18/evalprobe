"""Answer correctness: factual alignment with ground truth."""
from pydantic import BaseModel, Field
from ..types import EvalSample, MetricScore
from ..llm import call_json, DEFAULT_MODEL


CORRECTNESS_PROMPT = """Compare a candidate answer against a reference (ground truth) answer.

Step 1: Extract atomic factual claims from each. An atomic claim is a single, self-contained, verifiable statement of fact (e.g. "The Eiffel Tower is 330 meters tall"). It is NOT a comment about the comparison itself (e.g. "the date is missing").

Step 2: Classify each claim:
- TP (true positive): a factual claim present in BOTH the candidate and the reference (same meaning, even if worded differently or with different units like "330 metres" and "1,083 ft")
- FP (false positive): a factual claim made in the candidate that is NOT supported by the reference
- FN (false negative): a factual claim made in the reference that is NOT made by the candidate

Strict rules:
- Output ONLY atomic factual claims. Do NOT include meta-commentary, observations, or descriptions of what is missing or different.
- A unit conversion of the same fact (e.g. "330 metres" and "1,083 ft") is the SAME fact. Put it in TP, not FN.
- A more specific version of the same fact (e.g. "1889" vs "31 March 1889") is the SAME fact for TP. Do not penalize the candidate for being less specific.
- Each claim should be a short, self-contained sentence.

Question: {question}
Candidate answer: {candidate}
Reference answer: {reference}

Return ONLY valid JSON in this exact format:
{{
  "tp": ["fact 1", "fact 2"],
  "fp": ["fact 1"],
  "fn": ["fact 1"]
}}
"""


class _CorrectnessBreakdown(BaseModel):
    tp: list[str] = Field(default_factory=list)
    fp: list[str] = Field(default_factory=list)
    fn: list[str] = Field(default_factory=list)


def answer_correctness(
    sample: EvalSample, model: str = DEFAULT_MODEL
) -> MetricScore:
    """Answer correctness: F1 score of facts in candidate vs reference.

    Requires sample.ground_truth to be set. Score in [0.0, 1.0].

    Args:
        sample: An EvalSample with ground_truth set.
        model: LLM model name in litellm format.

    Returns:
        MetricScore with the F1 correctness score and breakdown details.
    """
    if not sample.ground_truth:
        return MetricScore(
            name="answer_correctness",
            score=0.0,
            details={"error": "ground_truth is required for correctness"},
        )

    breakdown = call_json(
        CORRECTNESS_PROMPT.format(
            question=sample.question,
            candidate=sample.answer,
            reference=sample.ground_truth,
        ),
        _CorrectnessBreakdown,
        model=model,
    )

    tp = len(breakdown.tp)
    fp = len(breakdown.fp)
    fn = len(breakdown.fn)

    if tp + fp + fn == 0:
        return MetricScore(
            name="answer_correctness",
            score=0.0,
            details={"error": "no facts classified"},
        )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return MetricScore(
        name="answer_correctness",
        score=f1,
        details={
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "tp": breakdown.tp,
            "fp": breakdown.fp,
            "fn": breakdown.fn,
        },
    )
