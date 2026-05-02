"""Tests for core data types."""
from ragmeter import EvalSample, MetricScore, EvalResult


def test_evalsample_minimal():
    s = EvalSample(question="q", answer="a")
    assert s.question == "q"
    assert s.answer == "a"
    assert s.contexts == []
    assert s.ground_truth is None


def test_evalsample_full():
    s = EvalSample(
        question="q",
        answer="a",
        contexts=["c1", "c2"],
        ground_truth="g",
    )
    assert len(s.contexts) == 2
    assert s.ground_truth == "g"


def test_metric_score():
    m = MetricScore(name="faithfulness", score=0.75)
    assert m.score == 0.75
    assert m.details == {}


def test_evalresult_serializes():
    s = EvalSample(question="q", answer="a")
    m = MetricScore(name="faithfulness", score=1.0, details={"x": 1})
    r = EvalResult(sample=s, scores=[m])
    blob = r.model_dump_json()
    assert "faithfulness" in blob
    assert '"score":1.0' in blob
