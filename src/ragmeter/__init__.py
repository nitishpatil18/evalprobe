"""ragmeter: lightweight, reliable RAG evaluation."""
from .types import EvalSample, EvalResult, MetricScore
from .evaluate import evaluate
from .metrics.faithfulness import faithfulness
from .llm import LLMError, DEFAULT_MODEL

__version__ = "0.1.0"

__all__ = [
    "evaluate",
    "faithfulness",
    "EvalSample",
    "EvalResult",
    "MetricScore",
    "LLMError",
    "DEFAULT_MODEL",
]
