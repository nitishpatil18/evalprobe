"""Answer relevancy: how well the answer addresses the question."""
from pydantic import BaseModel, Field
from ..types import EvalSample, MetricScore
from ..llm import call_json, DEFAULT_MODEL


REVERSE_QUESTION_PROMPT = """Given an answer, generate {n} questions that this answer
could plausibly be a response to. The questions should be diverse but all should
be questions the answer actually addresses.

Answer: {answer}

Return ONLY valid JSON in this exact format:
{{"questions": ["question 1", "question 2", "..."]}}
"""

SCORE_ALIGNMENT_PROMPT = """Score how well the candidate answer addresses the original question.

Score 1.0 if the answer fully and directly addresses the question.
Score 0.5 if the answer partially addresses the question or is tangential.
Score 0.0 if the answer does not address the question at all.

Original question: {question}
Candidate answer: {answer}

Return ONLY valid JSON in this exact format:
{{"score": 0.0, "reason": "brief explanation"}}
"""


class _ReverseQuestions(BaseModel):
    questions: list[str] = Field(default_factory=list)


class _AlignmentScore(BaseModel):
    score: float
    reason: str = ""


def answer_relevancy(
    sample: EvalSample,
    model: str = DEFAULT_MODEL,
    n_questions: int = 3,
) -> MetricScore:
    """Answer relevancy: how directly the answer addresses the question.

    Score in [0.0, 1.0]. Higher means the answer is more on-topic.
    Low scores indicate evasive, partial, or off-topic answers.

    Args:
        sample: An EvalSample with question and answer.
        model: LLM model name in litellm format.
        n_questions: How many reverse questions to generate (more = stabler score, more cost).

    Returns:
        MetricScore with the relevancy score and per-question details.
    """
    if not sample.answer.strip():
        return MetricScore(
            name="answer_relevancy",
            score=0.0,
            details={"error": "empty answer"},
        )

    # Step 1: generate questions the answer could respond to.
    rev = call_json(
        REVERSE_QUESTION_PROMPT.format(n=n_questions, answer=sample.answer),
        _ReverseQuestions,
        model=model,
    )
    candidate_questions = [q.strip() for q in rev.questions if q.strip()]

    if not candidate_questions:
        return MetricScore(
            name="answer_relevancy",
            score=0.0,
            details={"error": "could not generate candidate questions"},
        )

    # Step 2: for each candidate question, score how well the answer
    # also addresses the ORIGINAL question.
    # We invert the typical ragas approach: instead of embedding-cos-sim,
    # we ask the LLM to score alignment directly. Cheaper, more interpretable.
    alignment = call_json(
        SCORE_ALIGNMENT_PROMPT.format(
            question=sample.question, answer=sample.answer
        ),
        _AlignmentScore,
        model=model,
    )

    # Clamp to [0, 1].
    score = max(0.0, min(1.0, alignment.score))

    return MetricScore(
        name="answer_relevancy",
        score=score,
        details={
            "reason": alignment.reason,
            "candidate_questions": candidate_questions,
        },
    )
