"""Tests for the evaluate() entry point. No LLM calls."""
import pytest
from ragmeter import evaluate, EvalSample
from ragmeter.evaluate import METRIC_REGISTRY


def test_registry_has_three_metrics():
    assert set(METRIC_REGISTRY.keys()) == {
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
    }


def test_evaluate_unknown_metric_raises():
    s = EvalSample(question="q", answer="a")
    with pytest.raises(ValueError, match="Unknown metrics"):
        evaluate(s, metrics=["does_not_exist"])


def test_faithfulness_no_contexts_short_circuits():
    """faithfulness should return score=0 with an error detail, not call the LLM."""
    from ragmeter.metrics.faithfulness import faithfulness

    s = EvalSample(question="q", answer="a", contexts=[])
    score = faithfulness(s)
    assert score.score == 0.0
    assert "error" in score.details


def test_correctness_no_ground_truth_short_circuits():
    """answer_correctness should return score=0 if ground_truth missing."""
    from ragmeter.metrics.correctness import answer_correctness

    s = EvalSample(question="q", answer="a", ground_truth=None)
    score = answer_correctness(s)
    assert score.score == 0.0
    assert "error" in score.details


def test_relevancy_empty_answer_short_circuits():
    """answer_relevancy should return score=0 with error for empty answer."""
    from ragmeter.metrics.relevancy import answer_relevancy

    s = EvalSample(question="q", answer="   ")
    score = answer_relevancy(s)
    assert score.score == 0.0
    assert "error" in score.details
