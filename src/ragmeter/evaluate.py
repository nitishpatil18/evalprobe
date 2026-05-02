"""Top-level evaluate() entry point."""
from .types import EvalSample, EvalResult, MetricScore
from .metrics.faithfulness import faithfulness
from .llm import DEFAULT_MODEL


METRIC_REGISTRY = {
    "faithfulness": faithfulness,
}


def evaluate(
    sample: EvalSample,
    metrics: list[str] | None = None,
    model: str = DEFAULT_MODEL,
) -> EvalResult:
    """Evaluate a single sample with one or more metrics.

    Args:
        sample: The EvalSample to score.
        metrics: List of metric names. Defaults to all available metrics.
        model: LLM model name in litellm format.

    Returns:
        EvalResult with one MetricScore per requested metric.
    """
    metric_names = metrics or list(METRIC_REGISTRY.keys())

    unknown = [m for m in metric_names if m not in METRIC_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown metrics: {unknown}. "
            f"Available: {list(METRIC_REGISTRY.keys())}"
        )

    scores: list[MetricScore] = [
        METRIC_REGISTRY[name](sample, model=model) for name in metric_names
    ]
    return EvalResult(sample=sample, scores=scores)
