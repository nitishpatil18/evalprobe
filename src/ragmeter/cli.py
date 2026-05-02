"""Command-line interface for ragmeter."""
import argparse
import json
import sys
from pathlib import Path

from .evaluate import evaluate, METRIC_REGISTRY
from .types import EvalSample
from .llm import DEFAULT_MODEL, LLMError


def _load_samples(path: Path) -> list[EvalSample]:
    """Load samples from a JSONL file. One sample per line."""
    samples: list[EvalSample] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {i}: invalid JSON: {e}") from e
            try:
                samples.append(EvalSample.model_validate(data))
            except Exception as e:
                raise ValueError(f"Line {i}: invalid sample: {e}") from e
    return samples


def _format_summary(all_scores: dict[str, list[float]]) -> str:
    """Format a summary table of mean scores per metric."""
    lines = ["", "Summary:"]
    for name, scores in all_scores.items():
        if not scores:
            continue
        mean = sum(scores) / len(scores)
        lines.append(f"  {name:<22} mean = {mean:.3f}  (n={len(scores)})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ragmeter",
        description="Lightweight, reliable RAG evaluation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    eval_p = sub.add_parser("eval", help="Evaluate samples from a JSONL file.")
    eval_p.add_argument(
        "input", type=Path, help="Path to a JSONL file of samples."
    )
    eval_p.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Path to write JSONL results (default: stdout).",
    )
    eval_p.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model in litellm format (default: {DEFAULT_MODEL}).",
    )
    eval_p.add_argument(
        "--metrics",
        default=None,
        help=(
            "Comma-separated metric names. "
            f"Available: {','.join(METRIC_REGISTRY.keys())}. "
            "Default: all."
        ),
    )

    sub.add_parser("metrics", help="List available metrics.")

    args = parser.parse_args(argv)

    if args.command == "metrics":
        for name in METRIC_REGISTRY:
            print(name)
        return 0

    if args.command == "eval":
        if not args.input.exists():
            print(f"Error: input file not found: {args.input}", file=sys.stderr)
            return 1

        try:
            samples = _load_samples(args.input)
        except ValueError as e:
            print(f"Error reading input: {e}", file=sys.stderr)
            return 1

        if not samples:
            print("Error: input file has no samples.", file=sys.stderr)
            return 1

        metric_names = (
            [m.strip() for m in args.metrics.split(",")]
            if args.metrics
            else None
        )

        out_handle = (
            args.output.open("w", encoding="utf-8")
            if args.output
            else sys.stdout
        )

        all_scores: dict[str, list[float]] = {}
        try:
            for i, sample in enumerate(samples, start=1):
                print(f"[{i}/{len(samples)}] evaluating...", file=sys.stderr)
                try:
                    result = evaluate(
                        sample, metrics=metric_names, model=args.model
                    )
                except LLMError as e:
                    print(f"  LLM error: {e}", file=sys.stderr)
                    continue

                for s in result.scores:
                    all_scores.setdefault(s.name, []).append(s.score)
                    print(f"  {s.name}: {s.score:.3f}", file=sys.stderr)

                out_handle.write(result.model_dump_json() + "\n")
                out_handle.flush()
        finally:
            if args.output:
                out_handle.close()

        print(_format_summary(all_scores), file=sys.stderr)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
