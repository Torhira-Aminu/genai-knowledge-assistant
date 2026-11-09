"""Streamlit frontend for the Amazon Bedrock Converse demo."""
import logging
import os

import streamlit as st

from bedrock_client import (
    DEFAULT_MODEL_ID,
    DEFAULT_REGION,
    BedrockAssistant,
    BedrockInvocationError,
    ConfigurationError,
    InferenceSettings,
    history_from_chat,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("genai-assistant")

# Low-cost text models offered in the picker. Availability differs by account and
# region; scripts/check_bedrock.py shows what actually works for you.
MODEL_CHOICES = {
    "Amazon Nova Micro (lowest cost, text only)": "us.amazon.nova-micro-v1:0",
    "Amazon Nova Lite (low cost, multimodal)": "us.amazon.nova-lite-v1:0",
}

st.set_page_config(page_title="GenAI Knowledge Assistant", page_icon="💬")
st.title("GenAI Knowledge Assistant")
st.caption("Amazon Bedrock + Streamlit | Educational prototype — do not enter personal or patient data")

with st.sidebar:
    st.header("Model settings")
    env_model = os.getenv("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
    labels = list(MODEL_CHOICES)
    if env_model not in MODEL_CHOICES.values():
        labels.append(f"Custom (from BEDROCK_MODEL_ID): {env_model}")
        MODEL_CHOICES[labels[-1]] = env_model
    default_index = list(MODEL_CHOICES.values()).index(env_model)
    model_label = st.selectbox("Model", labels, index=default_index)
    region = st.text_input("AWS Region", os.getenv("AWS_REGION", DEFAULT_REGION))
    max_tokens = st.slider("Max response tokens", 50, 1000, 350, step=50,
                           help="Caps response length, which also caps cost per answer.")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, step=0.1,
                            help="Lower = more focused, higher = more varied.")
    if st.button("Clear conversation"):
        st.session_state.messages = []
    st.caption("Credentials come from your AWS profile / SSO login. They are never entered here.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask a non-sensitive question")
if prompt:
    history = history_from_chat(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    meta = None
    failed = True
    try:
        assistant = BedrockAssistant.from_environment(
            model_id=MODEL_CHOICES[model_label], region=region,
            settings=InferenceSettings(max_tokens=max_tokens, temperature=temperature))
        with st.spinner("Calling Amazon Bedrock..."):
            result = assistant.ask(prompt, history=history)
        answer = result.text
        failed = False
        meta = (f"{result.model_id} · {result.latency_ms} ms · "
                f"{result.input_tokens} in / {result.output_tokens} out tokens")
        if result.stop_reason == "max_tokens":
            answer += ("\n\n_⚠️ Answer cut off at the max-token limit. "
                       "Raise “Max response tokens” in the sidebar for a longer answer._")
        log.info("Bedrock OK model=%s latency_ms=%s", result.model_id, result.latency_ms)
    except BedrockInvocationError as exc:
        log.warning("Bedrock call failed code=%s", exc.code)
        answer = f"⚠️ {exc.user_message}"
    except (ConfigurationError, ValueError) as exc:
        answer = f"⚠️ {exc}"
    except Exception:  # last-resort guard so the UI never shows a stack trace
        log.exception("Unexpected error")
        answer = "⚠️ Unexpected error. See the terminal logs for details."

    st.session_state.messages.append({"role": "assistant", "content": answer, "error": failed})
    with st.chat_message("assistant"):
        st.markdown(answer)
        if meta:
            st.caption(meta)
