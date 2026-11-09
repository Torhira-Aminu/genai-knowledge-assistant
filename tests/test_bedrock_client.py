import unittest
from unittest.mock import MagicMock

from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

from bedrock_client import (
    BedrockAssistant,
    BedrockInvocationError,
    ConfigurationError,
    InferenceSettings,
    history_from_chat,
)


def ok_response(text="A concise answer."):
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "usage": {"inputTokens": 12, "outputTokens": 5},
        "metrics": {"latencyMs": 321},
        "stopReason": "end_turn",
    }


def client_error(code):
    return ClientError({"Error": {"Code": code, "Message": "secret internal detail"}}, "Converse")


class BedrockAssistantTests(unittest.TestCase):
    def test_converse_invocation_and_response(self):
        client = MagicMock()
        client.converse.return_value = ok_response()
        answer = BedrockAssistant(client, "example-model").ask("What is serverless?")
        self.assertEqual(answer.text, "A concise answer.")
        self.assertEqual((answer.input_tokens, answer.output_tokens, answer.latency_ms), (12, 5, 321))
        kwargs = client.converse.call_args.kwargs
        self.assertEqual(kwargs["modelId"], "example-model")
        self.assertEqual(kwargs["inferenceConfig"], {"maxTokens": 350, "temperature": 0.3, "topP": 0.9})
        self.assertEqual(kwargs["messages"][-1]["content"][0]["text"], "What is serverless?")

    def test_custom_inference_settings_sent(self):
        client = MagicMock()
        client.converse.return_value = ok_response()
        BedrockAssistant(client, "m", InferenceSettings(max_tokens=100, temperature=0.0)).ask("hi")
        self.assertEqual(client.converse.call_args.kwargs["inferenceConfig"]["maxTokens"], 100)

    def test_invalid_inference_settings_rejected(self):
        for bad in (InferenceSettings(max_tokens=0), InferenceSettings(temperature=1.5),
                    InferenceSettings(top_p=0)):
            with self.assertRaises(ConfigurationError):
                BedrockAssistant(MagicMock(), "m", bad)

    def test_model_id_required(self):
        with self.assertRaises(ConfigurationError):
            BedrockAssistant(MagicMock(), "  ")

    def test_empty_prompt_rejected(self):
        with self.assertRaises(ValueError):
            BedrockAssistant(MagicMock(), "model").ask("  ")

    def test_long_prompt_rejected_without_calling_aws(self):
        client = MagicMock()
        with self.assertRaises(ValueError):
            BedrockAssistant(client, "model").ask("x" * 4001)
        client.converse.assert_not_called()

    def test_missing_response_rejected(self):
        client = MagicMock()
        client.converse.return_value = {"output": {"message": {"content": []}}}
        with self.assertRaises(BedrockInvocationError):
            BedrockAssistant(client, "model").ask("hello")

    def test_access_denied_mapped_to_friendly_message(self):
        client = MagicMock()
        client.converse.side_effect = client_error("AccessDeniedException")
        with self.assertRaises(BedrockInvocationError) as ctx:
            BedrockAssistant(client, "model").ask("hello")
        self.assertEqual(ctx.exception.code, "AccessDeniedException")
        self.assertIn("IAM", ctx.exception.user_message)
        self.assertNotIn("secret internal detail", ctx.exception.user_message)

    def test_unknown_error_code_still_handled(self):
        client = MagicMock()
        client.converse.side_effect = client_error("SomethingNew")
        with self.assertRaises(BedrockInvocationError) as ctx:
            BedrockAssistant(client, "model").ask("hello")
        self.assertIn("SomethingNew", ctx.exception.user_message)

    def test_missing_credentials_handled(self):
        client = MagicMock()
        client.converse.side_effect = NoCredentialsError()
        with self.assertRaises(BedrockInvocationError) as ctx:
            BedrockAssistant(client, "model").ask("hello")
        self.assertEqual(ctx.exception.code, "NoCredentials")

    def test_network_error_handled(self):
        client = MagicMock()
        client.converse.side_effect = EndpointConnectionError(endpoint_url="https://example")
        with self.assertRaises(BedrockInvocationError):
            BedrockAssistant(client, "model").ask("hello")


class HistoryTests(unittest.TestCase):
    def test_history_starts_with_user_and_drops_failed_exchanges(self):
        chat = [
            {"role": "assistant", "content": "orphan"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "⚠️ error", "error": True},
        ]
        history = history_from_chat(chat)
        self.assertEqual([m["role"] for m in history], ["user", "assistant"])
        self.assertEqual(history[0]["content"][0]["text"], "q1")

    def test_history_is_trimmed(self):
        chat = []
        for i in range(20):
            chat += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
        self.assertEqual(len(history_from_chat(chat, max_turns=3)), 6)


if __name__ == "__main__":
    unittest.main()
