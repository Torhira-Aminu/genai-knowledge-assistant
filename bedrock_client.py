"""Small, testable wrapper around the Amazon Bedrock Converse API.

Credentials are never read or stored by this module. Boto3 resolves them from
the standard AWS credential chain (AWS SSO / `aws configure` profile, or
environment variables), so no secret ever appears in code or config files.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

# Cheap, text-only default. Many newer models are only callable in some regions
# through a cross-region *inference profile* ID (the "us." prefix). Run
# scripts/check_bedrock.py to confirm which IDs your account can actually use.
DEFAULT_MODEL_ID = "us.amazon.nova-micro-v1:0"
DEFAULT_REGION = "us-east-1"
MAX_PROMPT_CHARS = 4000

SYSTEM_PROMPT = (
    "You are a concise educational assistant. Do not request personal or patient data. "
    "For medical topics give general information, not diagnosis, and encourage qualified care."
)

# Maps Bedrock/AWS error codes to plain-English guidance that is safe to show
# in the UI (no stack traces, account IDs, or request details).
FRIENDLY_ERRORS = {
    "AccessDeniedException": "Your AWS identity is not allowed to call this model. "
    "Check the IAM policy (bedrock:InvokeModel) and that model access is enabled for your account.",
    "ResourceNotFoundException": "That model ID was not found in this region. "
    "Run scripts/check_bedrock.py to list usable model IDs.",
    "ValidationException": "Bedrock rejected the request. The model ID may need an inference-profile "
    "prefix (for example 'us.'), or an inference setting is out of range.",
    "ThrottlingException": "Bedrock is rate-limiting requests. Wait a few seconds and try again.",
    "ServiceQuotaExceededException": "Your account has hit a Bedrock quota for this model.",
    "ModelNotReadyException": "The model is not ready yet. Try again shortly.",
    "ModelTimeoutException": "The model took too long to respond. Try a shorter question.",
    "UnrecognizedClientException": "AWS did not recognize your credentials. Re-run `aws sso login` "
    "or `aws configure`.",
    "ExpiredTokenException": "Your AWS session has expired. Re-run `aws sso login`.",
}


class ConfigurationError(ValueError):
    """Required runtime configuration is missing or invalid."""


class BedrockInvocationError(RuntimeError):
    """A Bedrock call failed; `user_message` is safe to display."""

    def __init__(self, user_message: str, code: str = "Unknown"):
        super().__init__(user_message)
        self.user_message = user_message
        self.code = code


@dataclass
class Answer:
    text: str
    model_id: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    stop_reason: str | None = None


@dataclass
class InferenceSettings:
    max_tokens: int = 350
    temperature: float = 0.3
    top_p: float = 0.9
    extra: dict = field(default_factory=dict)

    def validate(self) -> None:
        if not 1 <= self.max_tokens <= 2000:
            raise ConfigurationError("max_tokens must be between 1 and 2000 for this demo.")
        if not 0.0 <= self.temperature <= 1.0:
            raise ConfigurationError("temperature must be between 0.0 and 1.0.")
        if not 0.0 < self.top_p <= 1.0:
            raise ConfigurationError("top_p must be greater than 0 and at most 1.0.")

    def to_api(self) -> dict:
        return {"maxTokens": self.max_tokens, "temperature": self.temperature, "topP": self.top_p}


class BedrockAssistant:
    def __init__(self, client, model_id: str, settings: InferenceSettings | None = None):
        if not model_id or not model_id.strip():
            raise ConfigurationError("BEDROCK_MODEL_ID is required.")
        self.client = client
        self.model_id = model_id.strip()
        self.settings = settings or InferenceSettings()
        self.settings.validate()

    @classmethod
    def from_environment(cls, model_id: str | None = None, region: str | None = None,
                         settings: InferenceSettings | None = None):
        import boto3  # imported here so unit tests never need AWS libraries configured
        from botocore.config import Config

        model_id = (model_id or os.getenv("BEDROCK_MODEL_ID") or DEFAULT_MODEL_ID).strip()
        region = (region or os.getenv("AWS_REGION") or DEFAULT_REGION).strip()
        profile = os.getenv("AWS_PROFILE") or None
        session = boto3.Session(region_name=region, profile_name=profile)
        config = Config(retries={"max_attempts": 3, "mode": "standard"},
                        connect_timeout=5, read_timeout=60)
        return cls(session.client("bedrock-runtime", config=config), model_id, settings)

    def ask(self, prompt: str, history: list[dict] | None = None) -> Answer:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"Prompt exceeds the {MAX_PROMPT_CHARS:,}-character demo limit.")

        messages = list(history or []) + [{"role": "user", "content": [{"text": prompt}]}]
        started = time.perf_counter()
        try:
            response = self.client.converse(
                modelId=self.model_id,
                system=[{"text": SYSTEM_PROMPT}],
                messages=messages,
                inferenceConfig=self.settings.to_api(),
            )
        except NoCredentialsError as exc:
            raise BedrockInvocationError(
                "No AWS credentials found. Run `aws sso login` or `aws configure` first.",
                "NoCredentials") from exc
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "Unknown")
            message = FRIENDLY_ERRORS.get(code, f"Bedrock request failed ({code}).")
            raise BedrockInvocationError(message, code) from exc
        except BotoCoreError as exc:
            raise BedrockInvocationError(
                "Could not reach Amazon Bedrock. Check your network connection and region.",
                type(exc).__name__) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        blocks = response.get("output", {}).get("message", {}).get("content", [])
        text = "\n".join(b["text"] for b in blocks if isinstance(b, dict) and b.get("text"))
        if not text.strip():
            raise BedrockInvocationError("Bedrock returned no text response.", "EmptyResponse")
        usage = response.get("usage", {})
        return Answer(
            text=text,
            model_id=self.model_id,
            input_tokens=usage.get("inputTokens"),
            output_tokens=usage.get("outputTokens"),
            latency_ms=response.get("metrics", {}).get("latencyMs", latency_ms),
            stop_reason=response.get("stopReason"),
        )


def history_from_chat(messages: list[dict], max_turns: int = 6) -> list[dict]:
    """Convert Streamlit chat history to Converse format, keeping only recent turns.

    Bedrock requires the conversation to start with a user turn and alternate roles,
    so the trimmed window is adjusted to begin on a user message.
    """
    clean = []
    for m in messages:
        if m.get("error"):
            # drop the failed exchange: the error note and the user turn that caused it
            if clean and clean[-1]["role"] == "user":
                clean.pop()
            continue
        clean.append(m)
    window = clean[-(max_turns * 2):]
    while window and window[0]["role"] != "user":
        window = window[1:]
    return [{"role": m["role"], "content": [{"text": m["content"]}]} for m in window]
