"""Core data types for evalprobe."""
from typing import Optional
from pydantic import BaseModel, Field


class EvalSample(BaseModel):
    """A single RAG evaluation sample."""
    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    ground_truth: Optional[str] = None


class MetricScore(BaseModel):
    """Score from a single metric."""
    name: str
    score: float
    details: dict = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Full evaluation result for one sample."""
    sample: EvalSample
    scores: list[MetricScore]
