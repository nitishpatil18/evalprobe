"""Answer correctness: factual alignment with ground truth."""
from pydantic import BaseModel, Field
from ..types import EvalSample, MetricScore
from ..llm import call_json, DEFAULT_MODEL


CORRECTNESS_PROMPT = """Compare a candidate answer against a reference (ground truth) answer.
Identify factual statements in each and classify them.

Definitions:
- TP (true positive): facts present in BOTH candidate and reference
- FP (false positive): facts in candidate but NOT in reference (hallucinations or extras)
- FN (false negative): facts in reference but missing from candidate

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
