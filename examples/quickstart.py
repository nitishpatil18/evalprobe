"""Smoke test: run faithfulness on a sample with a known hallucination."""
from dotenv import load_dotenv
from ragmeter import evaluate, EvalSample

load_dotenv()  # load OPENAI_API_KEY / GROQ_API_KEY from .env


sample = EvalSample(
    question="When did the Eiffel Tower open and how tall is it?",
    answer="The Eiffel Tower opened in 1889 and is 330 meters tall.",
    contexts=[
        "The Eiffel Tower is a wrought-iron lattice tower in Paris. "
        "It was completed in March 1889 for the World's Fair."
    ],
)

result = evaluate(sample, model="groq/llama-3.3-70b-versatile")

for score in result.scores:
    print(f"\n{score.name}: {score.score:.2f}")
    print("details:")
    for k, v in score.details.items():
        if k == "verifications":
            for v_item in v:
                mark = "supported" if v_item["supported"] else "unsupported"
                print(f"  [{mark}] {v_item['claim']}")
                print(f"     reason: {v_item['reason']}")
        else:
            print(f"  {k}: {v}")
