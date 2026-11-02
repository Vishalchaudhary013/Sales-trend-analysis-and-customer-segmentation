"""Streamlit entry point (landing page). Full content arrives in later milestones."""

import streamlit as st

from src import config

st.set_page_config(page_title=config.APP_TITLE, layout="wide")
st.title(config.APP_TITLE)
st.info(
    "Scaffolding in place (Milestone 1). Data, sales trends, segmentation and the churn "
    "model arrive in M2–M5."
)
