"""Streamlit frontend for Amazon Bedrock Converse demo."""
import logging
import streamlit as st
from bedrock_client import BedrockAssistant, ConfigurationError

st.set_page_config(page_title="GenAI Knowledge Assistant", page_icon="💬")
st.title("GenAI Knowledge Assistant")
st.caption("Amazon Bedrock + Streamlit | Educational prototype — no sensitive information")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask a non-sensitive question")
if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        assistant = BedrockAssistant.from_environment()
        with st.spinner("Calling Amazon Bedrock..."):
            answer = assistant.ask(prompt)
    except (ConfigurationError, Exception) as exc:
        logging.exception("Bedrock invocation failed (%s)", type(exc).__name__)
        answer = "Could not invoke Bedrock. Check model ID, region, credentials, and access. See application logs."
    st.session_state.messages.append({"role":"assistant","content":answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
