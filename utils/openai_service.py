import os
import json
import time
import inspect
from openai import OpenAI
from typing import Dict, List, Union

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

MODEL_NAME = "gpt-5-nano"

MAX_RETRIES = 3
RETRY_DELAY = 2


def get_ai_substitutions(meal_description: str,
                         allergens: List[str],
                         custom_rules: Dict[str, str] = {}) -> Dict[str, str]:
    """
    Get substitution suggestions from OpenAI for ingredients that need to be replaced.
    For single meal processing.
    """
    return get_batch_ai_substitutions([meal_description], allergens,
                                      custom_rules)[0]


def get_batch_ai_substitutions(
        meal_descriptions: List[str],
        allergens: List[str],
        custom_rules: Dict[str, str] = {}) -> List[Dict[str, str]]:
    """
    Get substitution suggestions from OpenAI for multiple meals at once.

    Args:
        meal_descriptions: List of meal descriptions to analyze
        allergens: List of allergens to avoid
        custom_rules: Dictionary of existing custom rules to follow

    Returns:
        List of dictionaries mapping original ingredients to their substitutions
    """
    if not meal_descriptions:
        return []

    if client is None:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. Please add it to your environment."
        )

    custom_rules_text = ""
    if custom_rules:
        custom_rules_text = "Follow these custom substitution rules first:\n"
        for original, replacement in custom_rules.items():
            custom_rules_text += f"- Replace '{original}' with '{replacement}'\n"

    meals_text = "\n".join([f"- {meal}" for meal in meal_descriptions])
    prompt = f"""Analyze these meals and suggest safe allergen-free substitutions for children in a school setting.

CRITICALLY IMPORTANT: These substitutions are for children with severe allergies in a school cafeteria. 
Even trace amounts of these allergens could cause life-threatening reactions.

*** EXTREMELY IMPORTANT: ONLY substitute ingredients that contain the SPECIFIC allergens listed below. DO NOT substitute ingredients for allergens that are not in the list. ***

{custom_rules_text}
Meals to analyze:
{meals_text}

Allergens to avoid: {', '.join(allergens)}

IMPORTANT REMINDERS ABOUT HIDDEN ALLERGENS:
- If "Egg Products" is listed: Replace ALL pancakes, waffles, muffins, enriched breads, and baked goods as these typically contain eggs
- If "Dairy" is listed: Replace ALL cheese, milk, yogurt, butter, cream cheese, and ice cream
- If "Fish" is listed: Replace ALL fish sticks, tuna, salmon, and seafood items
- If "Gluten" is listed: Replace ALL wheat, bread, pasta, crackers, and cereal products

ONLY replace ingredients that contain the SPECIFIC allergens listed above. DO NOT replace ingredients for any other allergens.
For example, if only "Fish" is listed as an allergen, do NOT replace dairy or gluten ingredients.

Replace any ingredients containing these specific allergens with safe alternatives that maintain nutritional value.

Where possible, substitutions should prefer common staple ingredients that would complete the meal, as opposed to specialty ingredients.  For example, here are some preferences: 
Noodles should be replaced with rice (Gluten)
Crackers with fruit (Gluten)
Most cereals with cheerios (Gluten)
Chicken patties with vegetarian patties (Vegetarian)
Chicken nuggets with vegetable nuggets (Vegetarian)
Turkey & Cheese sandwich with Turkey Sandwich (Dairy)
Turkey & Cheese sandwich with Egg patty sandwich (Vegetarian)


**** IMPORTANT: Copy the EXACT text of the original ingredient with any typos or abbreviations preserved in the response ****
Do not auto-correct or fix spelling errors in the original ingredients.

Important: Respond with a JSON array ONLY. Each element must be an object with the exact keys
"original" and "substitution". Do not wrap this array inside another object. Do not nest objects or arrays
any further. For example:
[
  {{
    "original": "original_ingredient_1",
    "substitution": "replacement_1"
  }},
  {{
    "original": "original_ingredient_2",
    "substitution": "replacement_2"
  }}
]

Only include ingredients that need to be substituted due to containing the SPECIFIC allergens listed.
Always return a direct mapping from original ingredient to replacement.
"""

    print("\n=== OpenAI API Request ===")
    print("Prompt:", prompt)
    print(f"Model: {MODEL_NAME}")

    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "original": {
                    "type": "string"
                },
                "substitution": {
                    "type": "string"
                }
            },
            "required": ["original", "substitution"],
            "additionalProperties": False
        }
    }

    def create_openai_request():
        """Create an OpenAI request using the Responses API."""

        if not hasattr(client, "responses"):
            raise AttributeError(
                "The configured OpenAI client does not support the responses API. "
                "Please upgrade the 'openai' package (>=1.57.0) so client.responses is available."
            )

        request_kwargs = {
            "model": MODEL_NAME,
            "max_output_tokens": 4000,
            "input": [
                {
                    "role": "system",
                    "content": "You are a dietary safety expert specializing in preventing severe allergic reactions in children. Your suggestions must be extremely cautious and prioritize safety above all else. ONLY suggest substitutions for SPECIFICALLY LISTED allergens. DO NOT substitute ingredients for allergens that weren't explicitly mentioned. For example, if only 'Fish' is listed as an allergen, do NOT replace dairy or gluten ingredients.\nKnow the hidden allergens: Eggs are in pancakes, waffles, muffins, and most baked goods. Dairy is in all cheese, milk, yogurt, and butter. Fish includes tuna and all seafood. Gluten is in all wheat, bread, pasta, and cereals.",
                },
                {"role": "user", "content": prompt},
            ],
        }

        # Older releases of the Responses API did not accept response_format. Detect
        # support dynamically to avoid runtime TypeErrors while still using the
        # schema when available.
        create_params = inspect.signature(client.responses.create).parameters
        if "response_format" in create_params:
            request_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "substitutions",
                    "schema": schema,
                    "strict": True,
                },
            }

        try:
            return client.responses.create(**request_kwargs)
        except TypeError as exc:
            if "response_format" in request_kwargs:
                # Retry without the schema parameter for environments running an
                # older SDK shape, but keep the strict JSON instructions in the
                # prompt so we still get structured output.
                request_kwargs.pop("response_format", None)
                return client.responses.create(**request_kwargs)
            raise exc

    def _coerce_json_value(value):
        """Convert a Responses json payload (value or callable) into a JSON string."""

        if callable(value):
            try:
                value = value()
            except TypeError:
                value = value({})

        if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
            return json.dumps(value)

        # Fallback to string conversion for any other objects (avoids TypeError for
        # methods or SDK-specific wrappers that aren't directly serializable).
        return json.dumps(str(value))

    def extract_message_content(response) -> str:
        """Extract text content from a Responses API payload."""

        if response and getattr(response, "output", None):
            output_items = response.output or []
            output_entry = output_items[0] if output_items else None
        else:
            output_entry = None

        if output_entry:
            if getattr(output_entry, "content", None):
                for content_part in output_entry.content:
                    if getattr(content_part, "text", None):
                        return content_part.text or ""
                    if getattr(content_part, "json", None) is not None:
                        return _coerce_json_value(content_part.json)
            if getattr(output_entry, "text", None):
                return output_entry.text or ""
            if getattr(output_entry, "json", None) is not None:
                return _coerce_json_value(output_entry.json)
        return ""

    response = None
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            response = create_openai_request()
            break
        except Exception as e:
            retry_count += 1
            error_msg = str(e)

            print(
                f"OpenAI API error (attempt {retry_count}/{MAX_RETRIES}): {error_msg}"
            )

            if retry_count >= MAX_RETRIES:
                print(
                    f"Maximum retries reached ({MAX_RETRIES}). Unable to get substitutions from OpenAI."
                )
                return [{} for _ in meal_descriptions]

            sleep_time = RETRY_DELAY * (2**(retry_count - 1))
            print(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)

    if response is None:
        return [{} for _ in meal_descriptions]

    try:
        print("\n=== OpenAI API Response ===")
        message_content = extract_message_content(response)
        message_content = message_content or ""
        print(message_content)
        print("===========================================\n")
        try:
            response_json = json.loads(message_content)

            substitutions_list = []

            # Expect a top-level array of substitution objects
            if isinstance(response_json, list):
                substitutions_list = [
                    item
                    for item in response_json
                    if isinstance(item, dict)
                    and "original" in item
                    and "substitution" in item
                ]

            # If the model wrapped the array in an object, try to unwrap common
            # variants gracefully without causing runtime errors.
            elif isinstance(response_json, dict):
                for key in ("meals", "substitutions", "items"):
                    possible_list = response_json.get(key)
                    if isinstance(possible_list, list):
                        substitutions_list = [
                            item
                            for item in possible_list
                            if isinstance(item, dict)
                            and "original" in item
                            and "substitution" in item
                        ]
                        break

            print(f"FINAL SUBSTITUTIONS LIST: {substitutions_list}")

            formatted_substitutions_dict = {
                item["original"]: item["substitution"]
                for item in substitutions_list
                if "original" in item and "substitution" in item
            }
            print(
                f"FORMATTED SUBSTITUTIONS LIST: {formatted_substitutions_dict}"
            )
            return [formatted_substitutions_dict]
        except json.JSONDecodeError as je:
            print(f"Error parsing JSON response: {str(je)}")
            return [{} for _ in meal_descriptions]
    except Exception as e:
        print(f"\nError processing AI substitutions: {str(e)}")
        return [{} for _ in meal_descriptions]
