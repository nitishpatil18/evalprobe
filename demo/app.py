"""Evalprobe demo on Hugging Face Spaces."""
import os
import streamlit as st
from evalprobe import evaluate, EvalSample, LLMError


DEFAULT_MODEL = "groq/llama-3.3-70b-versatile"

EXAMPLE = {
    "question": "When did the Eiffel Tower open and how tall is it?",
    "answer": "The Eiffel Tower opened in 1889 and is 330 meters tall.",
    "contexts": (
        "The Eiffel Tower is a wrought-iron lattice tower in Paris. "
        "It was completed in March 1889 for the World's Fair."
    ),
    "ground_truth": (
        "The Eiffel Tower opened to the public on 31 March 1889. "
        "It is approximately 330 metres (1,083 ft) tall."
    ),
}


st.set_page_config(
    page_title="evalprobe — RAG eval demo", page_icon="🎯", layout="centered"
)

st.title("evalprobe")
st.markdown(
    "**Lightweight, reliable RAG evaluation for solo AI builders.** "
    "Three metrics: faithfulness, relevancy, correctness. "
    "[GitHub](https://github.com/nitishpatil18/evalprobe) · "
    "[PyPI](https://pypi.org/project/evalprobe/)"
)

st.divider()

# Initialize session state with the example values so users see something useful
# from the first paint.
if "loaded" not in st.session_state:
    st.session_state.question = EXAMPLE["question"]
    st.session_state.answer = EXAMPLE["answer"]
    st.session_state.contexts = EXAMPLE["contexts"]
    st.session_state.ground_truth = EXAMPLE["ground_truth"]
    st.session_state.loaded = True

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Load example", use_container_width=True):
        st.session_state.question = EXAMPLE["question"]
        st.session_state.answer = EXAMPLE["answer"]
        st.session_state.contexts = EXAMPLE["contexts"]
        st.session_state.ground_truth = EXAMPLE["ground_truth"]
        st.rerun()
with col2:
    if st.button("Clear", use_container_width=True):
        st.session_state.question = ""
        st.session_state.answer = ""
        st.session_state.contexts = ""
        st.session_state.ground_truth = ""
        st.rerun()

question = st.text_area(
    "Question",
    key="question",
    height=70,
    placeholder="What did the user ask?",
)
answer = st.text_area(
    "Answer (from your RAG system)",
    key="answer",
    height=100,
    placeholder="The answer your RAG produced.",
)
contexts = st.text_area(
    "Retrieved contexts (one per line)",
    key="contexts",
    height=120,
    placeholder="Each line is a separate context chunk that was retrieved.",
)
ground_truth = st.text_area(
    "Ground truth (optional, needed for correctness)",
    key="ground_truth",
    height=80,
    placeholder="The known correct answer. Leave blank to skip correctness.",
)

run = st.button("Evaluate", type="primary", use_container_width=True)

if run:
    if not question.strip() or not answer.strip():
        st.error("Question and answer are required.")
        st.stop()

    if not os.environ.get("GROQ_API_KEY"):
        st.error(
            "GROQ_API_KEY is not configured on this Space. "
            "Run locally with your own key, or open an issue."
        )
        st.stop()

    contexts_list = [
        line.strip() for line in contexts.split("\n") if line.strip()
    ]
    sample = EvalSample(
        question=question.strip(),
        answer=answer.strip(),
        contexts=contexts_list,
        ground_truth=ground_truth.strip() or None,
    )

    metrics_to_run = ["faithfulness", "answer_relevancy"]
    if sample.ground_truth:
        metrics_to_run.append("answer_correctness")

    with st.spinner(f"Running {len(metrics_to_run)} metric(s)..."):
        try:
            result = evaluate(sample, metrics=metrics_to_run, model=DEFAULT_MODEL)
        except LLMError as e:
            st.error(f"LLM call failed: {e}")
            st.stop()

    st.divider()
    st.subheader("Results")

    cols = st.columns(len(result.scores))
    for col, score in zip(cols, result.scores):
        with col:
            st.metric(score.name, f"{score.score:.2f}")

    for score in result.scores:
        with st.expander(f"{score.name} — details", expanded=False):
            if score.name == "faithfulness":
                if "verifications" in score.details:
                    st.write(
                        f"**Claims supported:** "
                        f"{score.details.get('claims_supported', 0)} / "
                        f"{score.details.get('claims_total', 0)}"
                    )
                    for v in score.details["verifications"]:
                        mark = "✅" if v["supported"] else "❌"
                        st.markdown(f"{mark} **{v['claim']}**  \n_{v['reason']}_")
                else:
                    st.write(score.details)
            elif score.name == "answer_relevancy":
                st.write(score.details.get("reason", ""))
                cqs = score.details.get("candidate_questions", [])
                if cqs:
                    st.markdown("**Candidate questions the answer could respond to:**")
                    for q in cqs:
                        st.markdown(f"- {q}")
            elif score.name == "answer_correctness":
                p = score.details.get("precision", 0)
                r = score.details.get("recall", 0)
                st.write(f"**Precision:** {p:.2f}    **Recall:** {r:.2f}")
                if score.details.get("tp"):
                    st.markdown("**True positives (in both):**")
                    for f in score.details["tp"]:
                        st.markdown(f"- ✅ {f}")
                if score.details.get("fp"):
                    st.markdown("**False positives (in answer, not in ground truth):**")
                    for f in score.details["fp"]:
                        st.markdown(f"- ⚠️ {f}")
                if score.details.get("fn"):
                    st.markdown("**False negatives (missing from answer):**")
                    for f in score.details["fn"]:
                        st.markdown(f"- ❌ {f}")

    st.divider()
    st.markdown(
        "**Want to run this on your own data?**  \n"
        "`pip install evalprobe` — see [the docs](https://github.com/nitishpatil18/evalprobe)."
    )
