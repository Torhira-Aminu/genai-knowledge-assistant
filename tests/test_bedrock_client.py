import unittest
from unittest.mock import MagicMock
from bedrock_client import BedrockAssistant, ConfigurationError

class BedrockAssistantTests(unittest.TestCase):
    def test_converse_invocation_and_response(self):
        client = MagicMock()
        client.converse.return_value = {
            "output": {"message": {"content": [{"text": "A concise answer."}]}}
        }
        assistant = BedrockAssistant(client, "example-model")
        self.assertEqual(assistant.ask("What is serverless?"), "A concise answer.")
        self.assertEqual(client.converse.call_args.kwargs["modelId"], "example-model")

    def test_model_id_required(self):
        with self.assertRaises(ConfigurationError):
            BedrockAssistant(MagicMock(), "")

    def test_empty_prompt_rejected(self):
        with self.assertRaises(ValueError):
            BedrockAssistant(MagicMock(), "model").ask("  ")

    def test_missing_response_rejected(self):
        client = MagicMock()
        client.converse.return_value = {"output": {"message": {"content": []}}}
        with self.assertRaises(RuntimeError):
            BedrockAssistant(client, "model").ask("hello")

if __name__ == "__main__":
    unittest.main()
