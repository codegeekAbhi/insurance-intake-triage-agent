import streamlit as st
import pandas as pd
from groq import Groq

import triage_core as tc

st.set_page_config(page_title="Harper Triage Agent", layout="wide")

def load_css(path):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# API key: prefer Streamlit secrets (used on Streamlit Cloud), fall back to a
# manual sidebar entry for local testing. The key is never written to disk.
api_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
if not api_key:
    api_key = st.sidebar.text_input("Groq API key", type="password", help="Free at console.groq.com")

st.title("Harper Triage Agent")
st.caption("Commercial insurance intake triage for the automotive and transportation vertical")

if not api_key:
    st.warning("Add a Groq API key in the sidebar, or set GROQ_API_KEY in Streamlit secrets, to continue.")
    st.stop()

client = Groq(api_key=api_key)

tab1, tab2 = st.tabs(["Try it", "Eval harness"])

with tab1:
    st.write("Enter a business description and the agent will classify subtype, coverage need, urgency, and routing.")
    description = st.text_area(
        "Business description",
        placeholder="e.g. Family owned used car dealership in Ohio with 40 vehicles, dealer plates renew next week",
        height=100,
    )
    if st.button("Run triage", type="primary") and description.strip():
        with st.spinner("Classifying..."):
            result = tc.triage_with_guardrail(client, description)
        if "error" in result:
            st.error("Could not parse a response. Raw output below.")
            st.code(result.get("raw", ""))
        else:
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Subtype", result.get("subtype", "-"))
                col2.metric("Urgency", result.get("urgency", "-"))
                col3.metric("Routing", result.get("routing", "-"))
                col4.metric("Confidence", f"{result.get('confidence', 0):.2f}")
                st.write("**Coverage lines:** " + ", ".join(result.get("coverage_lines", [])))
                st.write("**Rationale:** " + result.get("rationale", ""))
                if result.get("guardrail_triggered"):
                    st.info("Confidence was at or below the threshold, routing was overridden to human.")

with tab2:
    st.write(f"{len(tc.EVAL_SET)} labeled scenarios in the golden set, across used car dealers, tow operators, freight carriers, and auto repair shops.")
    st.write("Running this calls the model once per scenario, twice, for the baseline prompt and the iterated prompt, spaced out to stay under Groq's free tier rate limit. Expect it to take two to three minutes.")

    if st.button("Run baseline vs iterated prompt comparison"):
        progress = st.progress(0.0, text="Running baseline prompt...")
        baseline_df, baseline_summary = tc.run_eval(
            client, tc.EVAL_SET, tc.SYSTEM_PROMPT_V1, label="baseline, v1 prompt",
            progress_callback=lambda p: progress.progress(p * 0.5, text="Running baseline prompt..."),
        )
        v2_df, v2_summary = tc.run_eval(
            client, tc.EVAL_SET, tc.SYSTEM_PROMPT_V2, label="v2 prompt, after iteration",
            progress_callback=lambda p: progress.progress(0.5 + p * 0.5, text="Running iterated prompt..."),
        )
        progress.empty()

        comparison = pd.DataFrame([baseline_summary, v2_summary]).set_index("label")
        with st.container(border=True):
            st.subheader("Baseline vs iterated prompt")
            st.dataframe(comparison.style.format("{:.2f}"), use_container_width=True)
            st.bar_chart(comparison.T)

            with st.expander("Baseline failure cases"):
                st.dataframe(
                    baseline_df[(baseline_df["subtype_correct"] == False) | (baseline_df["coverage_overlap"] < 1.0)],
                    use_container_width=True,
                )
