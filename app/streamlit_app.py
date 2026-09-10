from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fashion_style_ai_assistant.config import load_config  # noqa: E402
from fashion_style_ai_assistant.baseline.pipeline import FashionAssistant  # noqa: E402


@st.cache_resource
def load_assistant(config_path: str, data_dir: str) -> FashionAssistant:
    config = load_config(config_path)
    return FashionAssistant.from_data_dir(
        data_dir,
        max_features=config.max_features,
        ranking_config=config.ranking,
    )


st.set_page_config(page_title="Fashion Style AI Assistant", layout="wide")
st.title("Fashion Style AI Assistant")
st.caption("Retrieval + personalization + grounded product explanations + business metrics")

with st.sidebar:
    config_path = st.text_input("Config file", value="config.yaml")
    config = load_config(config_path)
    data_dir = st.text_input("Data directory", value=str(config.data_dir))
    user_id = st.text_input("User ID", value="u001")
    top_k = st.slider("Top K", min_value=1, max_value=10, value=config.default_top_k)
    candidate_k = st.slider("Candidate K", min_value=10, max_value=100, value=config.candidate_k)

query = st.text_input(
    "Style request",
    value="I need a relaxed light outfit for a beach weekend under 90 dollars",
)

if st.button("Recommend"):
    assistant = load_assistant(config_path, data_dir)
    response = assistant.recommend(query=query, user_id=user_id, top_k=top_k, candidate_k=candidate_k)
    if response.get("clarification_question"):
        st.info(response["clarification_question"])

    recs = response["recommendations"]
    st.subheader("Recommendations")
    for rec in recs:
        with st.expander(f"#{rec['rank']} {rec['name']} — ${rec['price']:.2f}", expanded=True):
            st.write("**Why**")
            for reason in rec["why"]:
                st.write(f"- {reason}")
            st.write("**Evidence**")
            evidence_df = pd.DataFrame(rec["evidence"])
            st.dataframe(evidence_df, use_container_width=True)
            st.write("**Business signals**")
            st.json(rec["business_signals"])

    st.subheader("Business metrics")
    st.json(response["business_metrics"])
    st.subheader("Business interpretation")
    st.json(response["business_interpretation"])
