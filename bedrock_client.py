"""Small, testable wrapper around the AWS Bedrock Converse API."""
import os
import boto3
from botocore.exceptions import ClientError, BotoCoreError

class ConfigurationError(ValueError):
    """Required runtime configuration is missing."""

SYSTEM_PROMPT = (
    "You are a concise educational assistant. Do not request personal or patient data. "
    "For medical topics give general information, not diagnosis, and encourage qualified care."
)

class BedrockAssistant:
    def __init__(self, client, model_id: str, max_tokens: int = 350):
        if not model_id:
            raise ConfigurationError("BEDROCK_MODEL_ID is required.")
        self.client = client
        self.model_id = model_id
        self.max_tokens = max_tokens

    @classmethod
    def from_environment(cls):
        model_id = os.getenv("BEDROCK_MODEL_ID", "").strip()
        region = os.getenv("AWS_REGION", "us-east-1")
        if not model_id:
            raise ConfigurationError("Set BEDROCK_MODEL_ID to a model available in your region.")
        session = boto3.Session(region_name=region)
        return cls(session.client("bedrock-runtime"), model_id)

    def ask(self, prompt: str) -> str:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        if len(prompt) > 4000:
            raise ValueError("Prompt exceeds the 4,000-character demo limit.")
        response = self.client.converse(
            modelId=self.model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": 0.3},
        )
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        result = "\n".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("text"))
        if not result.strip():
            raise RuntimeError("Bedrock returned no text response.")
        return result
