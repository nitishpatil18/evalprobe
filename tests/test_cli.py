"""Tests for the CLI argument parsing and file loading. No LLM calls."""
import json
import pytest
from evalprobe.cli import _load_samples, main


def test_load_samples_reads_jsonl(tmp_path):
    p = tmp_path / "samples.jsonl"
    p.write_text(
        '{"question": "q1", "answer": "a1"}\n'
        '{"question": "q2", "answer": "a2", "contexts": ["c"]}\n'
    )
    samples = _load_samples(p)
    assert len(samples) == 2
    assert samples[0].question == "q1"
    assert samples[1].contexts == ["c"]


def test_load_samples_skips_blank_lines(tmp_path):
    p = tmp_path / "samples.jsonl"
    p.write_text(
        '{"question": "q1", "answer": "a1"}\n'
        '\n'
        '{"question": "q2", "answer": "a2"}\n'
    )
    samples = _load_samples(p)
    assert len(samples) == 2


def test_load_samples_invalid_json_errors(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"not valid json\n')
    with pytest.raises(ValueError, match="Line 1"):
        _load_samples(p)


def test_cli_metrics_subcommand(capsys):
    rc = main(["metrics"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "faithfulness" in out
    assert "answer_relevancy" in out
    assert "answer_correctness" in out


def test_cli_missing_file_returns_error(capsys, tmp_path):
    missing = tmp_path / "nope.jsonl"
    rc = main(["eval", str(missing)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not found" in err
