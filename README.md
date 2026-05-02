# evalprobe

[![PyPI](https://img.shields.io/pypi/v/evalprobe)](https://pypi.org/project/evalprobe/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

**Lightweight, reliable RAG evaluation for solo AI builders.** Faithfulness, relevancy, and correctness in 5 lines. Works with any LLM via [litellm](https://github.com/BerriAI/litellm).

Other RAG eval libraries are heavy, brittle, or focused on enterprise. evalprobe is the smallest thing that actually works.

## Install

```bash
pip install evalprobe
```

## Use it

```python
from evalprobe import evaluate, EvalSample

sample = EvalSample(
    question="When did the Eiffel Tower open?",
    answer="It opened in 1889 and is 330 meters tall.",
    contexts=["The Eiffel Tower was completed in March 1889 for the World's Fair."],
    ground_truth="The Eiffel Tower opened on 31 March 1889.",
)

result = evaluate(sample, model="gpt-4o-mini")

for s in result.scores:
    print(f"{s.name}: {s.score:.2f}")
```

Output:
faithfulness: 0.50
answer_relevancy: 1.00
answer_correctness: 0.67

The height claim got flagged as unfaithful because the context didn't mention it. That's the whole point.

## CLI

Evaluate a JSONL file of samples and write results:

```bash
export OPENAI_API_KEY=sk-...

evalprobe eval samples.jsonl \
  --model gpt-4o-mini \
  --output results.jsonl
```

Each line of `samples.jsonl` is one sample:
```json
{"question": "...", "answer": "...", "contexts": ["..."], "ground_truth": "..."}
```

You get a per-metric mean summary on stderr and full per-sample results in `results.jsonl`.

## The metrics

- **faithfulness** — fraction of answer claims actually supported by the retrieved contexts. Catches hallucinations.
- **answer_relevancy** — how directly the answer addresses the question. Catches evasive or off-topic answers.
- **answer_correctness** — F1 score of facts in the answer vs. ground truth. Catches factual errors. Requires `ground_truth`.

All three return scores in `[0.0, 1.0]`. Higher is better.

## Any LLM, any provider

evalprobe uses [litellm](https://github.com/BerriAI/litellm) under the hood, so any model name litellm understands works:

```python
evaluate(sample, model="gpt-4o-mini")
evaluate(sample, model="groq/llama-3.3-70b-versatile")
evaluate(sample, model="anthropic/claude-3-5-sonnet-latest")
evaluate(sample, model="ollama/llama3.1")
```

Set the corresponding `OPENAI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, etc.

## Why evalprobe

| | evalprobe | ragas | langsmith |
|---|---|---|---|
| Install size | small | heavy | heavy |
| LLM provider | any (litellm) | partial | partial |
| Errors | clear messages | sometimes silent | n/a |
| Hosted dashboard | _coming_ | no | yes (paid) |
| Built for | solo devs | research/teams | enterprise |

## Status

Pre-alpha. Things will change. Star to follow.

## License

MIT