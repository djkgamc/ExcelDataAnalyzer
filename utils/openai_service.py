import os
import json
import time
from openai import OpenAI
from typing import Dict, List, Union

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set. Please add it to your environment variables."
    )

client = OpenAI(api_key=api_key)

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

Important: Respond with a JSON object in this format ONLY: 
{{
  "meals": [
    {{
      "original": "original_ingredient_1",
      "substitution": "replacement_1"
    }},
    {{
      "original": "original_ingredient_2",
      "substitution": "replacement_2"
    }}
  ]
}}

Only include ingredients that need to be substituted due to containing the SPECIFIC allergens listed.
Always use 'meals' as the key (not 'meal').
Always return a direct mapping from original ingredient to replacement.
Do not use nested objects or arrays for the substitutions.
"""

    print("\n=== OpenAI API Request ===")
    print("Prompt:", prompt)
    print("Model: o4-mini")

    schema = {
        "type": "object",
        "properties": {
            "meals": {
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
        },
        "required": ["meals"],
        "additionalProperties": False
    }

    response = None
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            response = client.responses.create(
                model="o4-mini",
                max_output_tokens=16000,
                instructions=
                "You are a dietary safety expert specializing in preventing severe allergic reactions in children. Your suggestions must be extremely cautious and prioritize safety above all else. ONLY suggest substitutions for SPECIFICALLY LISTED allergens. DO NOT substitute ingredients for allergens that weren't explicitly mentioned. For example, if only 'Fish' is listed as an allergen, do NOT replace dairy or gluten ingredients.\nKnow the hidden allergens: Eggs are in pancakes, waffles, muffins, and most baked goods. Dairy is in all cheese, milk, yogurt, and butter. Fish includes tuna and all seafood. Gluten is in all wheat, bread, pasta, and cereals.",
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "substitutions",
                        "schema": schema,
                        "strict": True
                    }
                },
            )
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
        print(response.output_text)
        print("===========================================\n")
        try:
            response_json = json.loads(response.output_text)

            substitutions_list = []

            if isinstance(response_json, dict) and 'meals' in response_json:
                meals_array = response_json['meals']
                print(f"MEALS ARRAY FROM API: {meals_array}")

                substitutions_list = []
                for meal in meals_array:
                    if isinstance(
                            meal, dict
                    ) and 'original' in meal and 'substitution' in meal:
                        substitutions_list.append(meal)

                while len(substitutions_list) < len(meal_descriptions):
                    substitutions_list.append({})

                print(f"FINAL SUBSTITUTIONS LIST: {substitutions_list}")

            else:
                substitutions_list = [{}
                                      for _ in range(len(meal_descriptions))]

            formatted_substitutions_dict = {
                item['original']: item['substitution']
                for item in substitutions_list
                if 'original' in item and 'substitution' in item
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
