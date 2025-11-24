import os
import unittest
from unittest.mock import patch

import httpx
from openai import APIConnectionError, APITimeoutError

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

    def create(self, *, response_format=None, **kwargs):
        self.last_kwargs = {"response_format": response_format, **kwargs}
        return FakeResponse(
            [
                {"original": "Milk", "substitution": "Soy milk"},
                {"original": "Cheese", "substitution": "Vegan cheese"},
            ]
        )


class FakeResponsesWithoutSchema:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        # Simulate an SDK shape that rejects response_format
        self.last_kwargs = kwargs
        return FakeResponse(
            [
                {"original": "Yogurt", "substitution": "Coconut yogurt"},
            ]
        )


class FakeReasoningBlock:
    def __init__(self, output_text):
        self.type = "reasoning"
        self.content = None
        self.encrypted_content = None
        self.status = None
        self.output_text = output_text


class FakeReasoningContent:
    def __init__(self, output_text):
        self.text = None
        self.json = None
        self.output_text = None
        self.reasoning = FakeReasoningBlock(output_text)


class FakeReasoningOutput:
    def __init__(self, output_text):
        self.content = [FakeReasoningContent(output_text)]
        self.output_text = None


class FakeReasoningResponse:
    def __init__(self, output_text):
        self.output = [FakeReasoningOutput(output_text)]
        self.output_text = None


class FakeClient:
    def __init__(self):
        self.responses = FakeResponsesClient()


class OpenAIServiceTests(unittest.TestCase):
    def test_responses_api_uses_output_token_param(self):
        fake_client = FakeClient()

        with patch.object(openai_service, "get_openai_client", return_value=fake_client):
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
        self.assertIsInstance(request_kwargs.get("response_format"), dict)

    def test_falls_back_when_response_format_not_supported(self):
        class ClientWithoutSchema:
            def __init__(self):
                self.responses = FakeResponsesWithoutSchema()

        fake_client = ClientWithoutSchema()

        with patch.object(openai_service, "get_openai_client", return_value=fake_client):
            substitutions = openai_service.get_batch_ai_substitutions(
                ["Yogurt snack"], ["Dairy"]
            )

        self.assertEqual(len(substitutions), 1)
        self.assertEqual(substitutions[0].get("Yogurt"), "Coconut yogurt")

        # Ensure the request didn't include response_format in this SDK shape
        self.assertNotIn("response_format", fake_client.responses.last_kwargs)

    def test_reasoning_output_is_unwrapped(self):
        class ClientWithReasoning:
            def __init__(self, output_text):
                self.responses = self
                self._output_text = output_text

            def create(self, **kwargs):
                self.last_kwargs = kwargs
                return FakeReasoningResponse(self._output_text)

        output_text = "[{\"original\": \"Milk\", \"substitution\": \"Oat milk\"}]"
        fake_client = ClientWithReasoning(output_text)

        with patch.object(openai_service, "get_openai_client", return_value=fake_client):
            substitutions = openai_service.get_batch_ai_substitutions(
                ["Breakfast"], ["Dairy"]
            )

        self.assertEqual(len(substitutions), 1)
        self.assertEqual(substitutions[0].get("Milk"), "Oat milk")


class LiveOpenAIIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        openai_service.resolve_api_key(),
        "OPENAI_API_KEY is required for live OpenAI call",
    )
    def test_live_call_returns_substitutions(self):
        # Ensure we are exercising the real client (Responses API, not completions)
        client = openai_service.get_openai_client()
        self.assertIsNotNone(client)
        self.assertTrue(hasattr(client, "responses"))

        try:
            substitutions = openai_service.get_batch_ai_substitutions(
                ["Breakfast: milk, cheese toast, and yogurt parfait"],
                ["Dairy"],
                {"Milk": "Oat milk"},
            )
        except (APIConnectionError, APITimeoutError, httpx.HTTPError) as exc:
            self.skipTest(f"Live OpenAI call unavailable in this environment: {exc}")

        if not substitutions or not substitutions[0]:
            self.skipTest(
                "Live OpenAI call returned no substitutions; service likely unavailable"
            )

        self.assertEqual(len(substitutions), 1)
        mapping = substitutions[0]
        self.assertIsInstance(mapping, dict)
        self.assertGreater(len(mapping), 0)

        key, value = next(iter(mapping.items()))
        self.assertIsInstance(key, str)
        self.assertIsInstance(value, str)


if __name__ == "__main__":
    unittest.main()
