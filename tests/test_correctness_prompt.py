"""Smoke tests for the correctness prompt structure."""
from evalprobe.metrics.correctness import CORRECTNESS_PROMPT


def test_prompt_forbids_meta_commentary():
    """Prompt must instruct the LLM to output only atomic facts."""
    text = CORRECTNESS_PROMPT.lower()
    assert "atomic" in text
    assert "meta-commentary" in text or "commentary" in text


def test_prompt_handles_unit_equivalence():
    """Prompt must instruct that unit conversions are the same fact."""
    text = CORRECTNESS_PROMPT.lower()
    assert "unit" in text


def test_prompt_handles_specificity():
    """Prompt must instruct that more-specific versions are the same fact."""
    text = CORRECTNESS_PROMPT.lower()
    assert "specific" in text


def test_prompt_has_required_format_template():
    """Prompt must request the tp/fp/fn JSON shape."""
    assert '"tp"' in CORRECTNESS_PROMPT
    assert '"fp"' in CORRECTNESS_PROMPT
    assert '"fn"' in CORRECTNESS_PROMPT
