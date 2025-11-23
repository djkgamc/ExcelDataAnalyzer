import unittest
from unittest.mock import patch

from utils import openai_service


class FakeContent:
    def __init__(self, json_value):
        self.json = json_value
        self.text = None


class FakeOutput:
    def __init__(self, json_value):
        self.content = [FakeContent(json_value)]


class FakeResponse:
    def __init__(self, json_value):
        self.output = [FakeOutput(json_value)]


class FakeResponsesClient:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return FakeResponse(
            {
                "meals": [
                    {"original": "Milk", "substitution": "Soy milk"},
                    {"original": "Cheese", "substitution": "Vegan cheese"},
                ]
            }
        )


class FakeClient:
    def __init__(self):
        self.responses = FakeResponsesClient()


class OpenAIServiceTests(unittest.TestCase):
    def test_responses_api_uses_output_token_param(self):
        fake_client = FakeClient()

        with patch.object(openai_service, "client", fake_client):
            substitutions = openai_service.get_batch_ai_substitutions(
                ["Meal description"], ["Dairy"]
            )

        self.assertEqual(len(substitutions), 1)
        self.assertIn("Milk", substitutions[0])
        self.assertEqual(substitutions[0]["Milk"], "Soy milk")

        # Verify the Responses API call used the correct keyword argument
        request_kwargs = fake_client.responses.last_kwargs
        self.assertIsNotNone(request_kwargs)
        self.assertEqual(request_kwargs.get("max_output_tokens"), 4000)
        self.assertNotIn("max_completion_tokens", request_kwargs)


if __name__ == "__main__":
    unittest.main()
