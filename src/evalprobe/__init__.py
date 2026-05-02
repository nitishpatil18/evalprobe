"""evalprobe: lightweight, reliable RAG evaluation."""
from .types import EvalSample, EvalResult, MetricScore
from .evaluate import evaluate
from .metrics.faithfulness import faithfulness
from .metrics.relevancy import answer_relevancy
from .metrics.correctness import answer_correctness
from .llm import LLMError, DEFAULT_MODEL

__version__ = "0.1.1"

__all__ = [
    "evaluate",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "EvalSample",
    "EvalResult",
    "MetricScore",
    "LLMError",
    "DEFAULT_MODEL",
]
