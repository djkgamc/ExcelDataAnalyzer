import os
import json
import time
from typing import Dict, List, Optional
import re

import httpx
from openai import OpenAI

_client_cache = None
_client_api_key = None

MODEL_NAME = "gpt-5.1"
# MODEL_NAME = "gpt-4.1"

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


def resolve_api_key() -> Optional[str]:
    """Resolve the OpenAI API key from the environment using flexible keys."""

    for key_name in (
        "OPENAI_API_KEY",
        "openai_api_key",
        "OPENAI_API_KEY_ENV",
        "openai_api_key_env",
    ):
        value = os.environ.get(key_name)
        if value:
            return value
    return None


def get_openai_client() -> OpenAI:
    """Return a cached OpenAI client using the currently configured API key."""

    global _client_cache, _client_api_key

    api_key = resolve_api_key()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. Please add it to your environment."
        )

    if _client_cache is not None and api_key == _client_api_key:
        return _client_cache

    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    http_client = build_http_client()
    if http_client:
        client_kwargs["http_client"] = http_client

    _client_cache = OpenAI(**client_kwargs)
    _client_api_key = api_key
    return _client_cache


def build_http_client() -> Optional[httpx.Client]:
    """Create an httpx client that honors optional proxy overrides."""

    # Allow explicit disabling of proxies when they block outbound calls.
    if os.environ.get("OPENAI_DISABLE_PROXY", "").lower() in {"1", "true", "yes"}:
        proxies = None
    else:
        proxies = (
            os.environ.get("OPENAI_HTTP_PROXY")
            or os.environ.get("OPENAI_HTTPS_PROXY")
            or os.environ.get("HTTPS_PROXY")
            or os.environ.get("HTTP_PROXY")
        )

    try:
        return httpx.Client(proxies=proxies, transport=httpx.HTTPTransport(retries=2))
    except Exception:
        # If proxy configuration is invalid, fall back to default client behavior.
        return None


def get_batch_ai_substitutions(
        meal_descriptions: List[str],
        allergens: List[str],
        custom_rules: Dict[str, str] = {},
        progress_callback=None) -> List[Dict[str, str]]:
    """
    Get substitution suggestions from OpenAI for multiple meals at once.

    Args:
        meal_descriptions: List of meal descriptions to analyze
        allergens: List of allergens to avoid
        custom_rules: Dictionary of existing custom rules to follow
        progress_callback: Optional callback function(text: str) called with reasoning text chunks during streaming

    Returns:
        List of dictionaries mapping original ingredients to their substitutions
    """
    if not meal_descriptions:
        return []

    client = get_openai_client()

    # Build a normalized, deduplicated ingredient list for the model to choose from.
    def _normalize_display(text: str) -> str:
        return " ".join(str(text).split())

    def _extract_meal_parts(cell_text: str) -> List[str]:
        """
        Extract text for each meal marker (B:, L:, S:) separately, then split
        by commas within each part. This avoids tokens that span markers.
        """
        parts: List[str] = []
        text = str(cell_text)
        # Regex: capture the content after a marker until the next marker or end
        pattern = re.compile(r"^[ \t]*([BLS])\s*:\s*(.*?)(?=^[ \t]*[BLS]\s*:\s*|\Z)", re.IGNORECASE | re.MULTILINE | re.DOTALL)
        for m in pattern.finditer(text):
            content = m.group(2)
            # Split within this part on commas only
            for token in content.split(","):
                tok = token.strip()
                if tok:
                    parts.append(tok)
        # As a fallback, if no markers matched, split whole cell on commas
        if not parts:
            for token in text.split(","):
                tok = token.strip()
                if tok:
                    parts.append(tok)
        return parts

    ingredient_norm_to_id: Dict[str, str] = {}
    ingredient_id_to_raw: Dict[str, str] = {}
    ingredient_list_for_model: List[Dict[str, str]] = []

    next_id = 1
    for cell_text in meal_descriptions:
        if not cell_text:
            continue
        # Extract tokens within each meal part
        for raw_token in _extract_meal_parts(cell_text):
            if not raw_token:
                continue
            norm_key = _normalize_display(raw_token).lower()
            if norm_key not in ingredient_norm_to_id:
                ing_id = f"ing_{next_id}"
                next_id += 1
                ingredient_norm_to_id[norm_key] = ing_id
                ingredient_id_to_raw[ing_id] = raw_token
                ingredient_list_for_model.append({
                    "id": ing_id,
                    "name": _normalize_display(raw_token),
                })

    prompt_payload = {
        "task": "allergen_substitutions",
        "audience": "children with severe allergies in a school cafeteria",
        "constraints": {
            "allergens_to_avoid": allergens or [],
            "hidden_allergen_reminders": [
                "Eggs: pancakes, waffles, muffins, enriched breads, baked goods",
                "Dairy: cheese, milk, yogurt, butter, cream cheese, ice cream",
                "Fish: fish sticks, tuna, salmon, seafood items",
                "Gluten: wheat, bread, pasta, crackers, cereal products",
            ],
            "only_replace_listed_allergens": True,
            "preferences": [
                "Noodles -> rice (Gluten)",
                "Crackers -> fruit (Gluten)",
                "Most cereals -> cheerios (Gluten)",
                "Chicken patties -> vegetarian patties (Vegetarian)",
                "Chicken nuggets -> vegetable nuggets (Vegetarian)",
                "Turkey & Cheese -> Turkey Sandwich (Dairy)",
                "Turkey & Cheese -> Egg patty sandwich (Vegetarian)",
            ],
            "substitution_style": {
                "must_be_short_menu_label": True,
                "max_words": 8,
                "forbid_sentences_or_instructions": True,
                "forbid_phrases": ["instead of", "served with", "serve with", "prepared", "using", "over"],
                "forbid_ids_in_text": ["ing_"],
                "forbid_punctuation": [",", "."],
                "allowed_examples": [
                    "Soy milk",
                    "Fresh fruit",
                    "Brown rice",
                    "Gluten-free bread",
                    "Rice Chex",
                    "Corn tortilla",
                    "Sun butter",
                    "Mashed potatoes (no dairy)"
                ],
                "disallowed_examples": [
                    "Fresh fruit instead of crackers",
                    "WGR Cheerios (ing_30) served with fruit",
                    "Serve turkey with salad"
                ]
            },
            "allergen_keyword_hints": {
                "gluten_like": ["wgr", "wheat", "bun", "bread", "roll", "noodle", "pasta", "cracker", "graham", "pretzel"],
                "dairy_like": ["milk", "cheese", "yogurt", "cream", "mac and cheese"]
            },
            "deduplicate_by_id": True
        },
        "custom_rules": [
            {"original": k, "replacement": v} for k, v in (custom_rules or {}).items()
        ],
        "ingredients": ingredient_list_for_model,
        "output_requirement": {
            "format": "json_array",
            "schema": {"id": "string", "substitution": "string", "original": "string (optional)"},
            "notes": [
                "Return ONLY items that contain the SPECIFIC listed allergens.",
                "Select only from provided ingredients by id (the 'id' field).",
                "The 'substitution' must be a concise, menu-ready name (e.g., 'Soy milk', 'Fresh fruit', 'Brown rice', 'Gluten-free bread').",
                "Do NOT include sentences, 'instead of', serving instructions, commas/periods, or any (ing_XX) references inside 'substitution'.",
                "You may stream a few short progress lines first.",
                "When you are ready to answer, print exactly the single line: ===JSON===",
                "Immediately after that line, output ONE JSON array and NOTHING ELSE after the closing ']'.",
                "If there are no substitutions, output [] and NOTHING ELSE.",
            ],
        },
    }

    prompt = (
        "Analyze the ingredient list and propose safe allergen-free substitutions for a school cafeteria. "
        "Follow all safety and constraint rules.\n"
        "1) Stream a few short progress lines as you think.\n"
        "2) Then print exactly the line ===JSON===\n"
        "3) Then output ONLY a single JSON array of objects with keys {\"id\",\"substitution\"} and optionally {\"original\"}. "
        "Keep each substitution ≤ 8 words, no sentences/instructions.\n"
        "If none apply, output [] after the marker.\n"
        "Example after the marker:\n"
        "[{\"id\":\"ing_1\",\"original\":\"Milk\",\"substitution\":\"Soy milk\"},{\"id\":\"ing_23\",\"substitution\":\"Brown rice\"}]\n\n"
        f"DATA:\n{json.dumps(prompt_payload, ensure_ascii=False)}"
    )

    print("\n=== OpenAI API Request ===")
    print("Prompt:", prompt)
    print(f"Model: {MODEL_NAME}")

    # No strict schema here; we stream JSON text and validate after

    def create_openai_request():
        """Create an OpenAI request using the Responses API."""

        if not hasattr(client, "responses"):
            raise AttributeError(
                "The configured OpenAI client does not support the responses API. "
                "Please upgrade the 'openai' package (>=1.57.0) so client.responses is available."
            )

        # Responses API uses 'text' parameter for structured outputs, not 'response_format'
        # Set max_output_tokens very high to ensure we never hit the limit
        # Enable streaming to show reasoning text in real-time
        return client.responses.create(
            model=MODEL_NAME,
            max_output_tokens=32000,  # Very high limit to handle reasoning + large JSON output
            input=[
                {
                    "role": "system",
                    "content": "You are a dietary safety expert specializing in preventing severe allergic reactions in children. Your suggestions must be extremely cautious and prioritize safety above all else. ONLY suggest substitutions for SPECIFICALLY LISTED allergens. DO NOT substitute ingredients for allergens that weren't explicitly mentioned. For example, if only 'Fish' is listed as an allergen, do NOT replace dairy or gluten ingredients.\nKnow the hidden allergens: Eggs are in pancakes, waffles, muffins, and most baked goods. Dairy is in all cheese, milk, yogurt, and butter. Fish includes tuna and all seafood. Gluten is in all wheat, bread, pasta, and cereals.",
                },
                {"role": "user", "content": prompt},
            ],
            reasoning={"effort": "medium", "summary": "auto"},
            stream=True,  # Enable streaming to show reasoning in real-time
        )

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

    def extract_message_content(response, prefer_json: bool = True) -> str:
        """Extract text content from a Responses API payload.
        
        Args:
            response: The Responses API response object
            prefer_json: If True, prioritize JSON content over text/reasoning (for structured outputs)
        """

        json_parts: List[str] = []
        text_parts: List[str] = []

        def add_json_part(value):
            if value is None:
                return
            json_str = _coerce_json_value(value)
            if json_str and json_str.strip():
                json_parts.append(json_str)

        def add_text_part(value):
            if value is None:
                return
            if isinstance(value, (dict, list, str, int, float, bool)):
                text_value = value if isinstance(value, str) else json.dumps(value)
            else:
                text_value = str(value)
            if text_value and text_value.strip():
                text_parts.append(text_value)

        if response is None:
            return ""

        # Check if response is incomplete - if so, we might need to handle it differently
        response_status = getattr(response, "status", None)
        if response_status == "incomplete":
            incomplete_details = getattr(response, "incomplete_details", None)
            if incomplete_details and getattr(incomplete_details, "reason", None) == "max_output_tokens":
                # Response hit token limit - check if there's any partial output we can extract
                # The actual output might be in response.text or we need to retry with more tokens
                pass

        # Check response.text directly - it might contain the actual output content
        response_text = getattr(response, "text", None)
        if response_text:
            # response.text might be a config object, check if it has actual content
            if isinstance(response_text, dict):
                # Look for actual text content in the text config
                text_content = response_text.get("content") or response_text.get("output") or response_text.get("text")
                if text_content:
                    if prefer_json:
                        try:
                            json.loads(text_content)
                            json_parts.append(text_content)
                        except (json.JSONDecodeError, TypeError):
                            if not json_parts:
                                text_parts.append(text_content)
                    else:
                        text_parts.append(text_content)
            elif isinstance(response_text, str):
                if prefer_json:
                    try:
                        json.loads(response_text)
                        json_parts.append(response_text)
                    except (json.JSONDecodeError, TypeError):
                        if not json_parts:
                            text_parts.append(response_text)
                else:
                    text_parts.append(response_text)

        # Prefer any aggregated helpers the SDK provides.
        if getattr(response, "output_text", None):
            if prefer_json:
                # Try to parse as JSON first
                try:
                    json.loads(response.output_text)
                    add_json_part(response.output_text)
                except (json.JSONDecodeError, TypeError):
                    add_text_part(response.output_text)
            else:
                add_text_part(response.output_text)

        outputs = getattr(response, "output", None) or []
        for output_entry in outputs:
            # Skip reasoning-only entries that don't have actual content
            entry_type = getattr(output_entry, "type", None)
            if entry_type == "reasoning":
                # Check if reasoning entry has actual output content
                reasoning_output = getattr(output_entry, "output", None)
                if reasoning_output:
                    # Process reasoning output recursively
                    if isinstance(reasoning_output, list):
                        for item in reasoning_output:
                            if getattr(item, "json", None) is not None:
                                add_json_part(item.json)
                            elif getattr(item, "text", None):
                                text_val = item.text
                                if prefer_json:
                                    try:
                                        json.loads(text_val)
                                        add_json_part(text_val)
                                    except (json.JSONDecodeError, TypeError):
                                        if not json_parts:
                                            add_text_part(text_val)
                                else:
                                    add_text_part(text_val)
                continue
            
            if getattr(output_entry, "output_text", None):
                if prefer_json:
                    try:
                        json.loads(output_entry.output_text)
                        add_json_part(output_entry.output_text)
                    except (json.JSONDecodeError, TypeError):
                        add_text_part(output_entry.output_text)
                else:
                    add_text_part(output_entry.output_text)

            contents = getattr(output_entry, "content", None) or []
            for content_part in contents:
                # Skip reasoning blocks - they don't contain the actual output
                content_type = getattr(content_part, "type", None)
                if content_type == "reasoning":
                    # Reasoning blocks may contain the actual output, check for it
                    reasoning_output = getattr(content_part, "output", None)
                    if reasoning_output:
                        # Recursively check reasoning output for actual content
                        if isinstance(reasoning_output, list):
                            for reasoning_item in reasoning_output:
                                if getattr(reasoning_item, "json", None) is not None:
                                    add_json_part(reasoning_item.json)
                                elif getattr(reasoning_item, "text", None):
                                    if prefer_json:
                                        try:
                                            json.loads(reasoning_item.text)
                                            add_json_part(reasoning_item.text)
                                        except (json.JSONDecodeError, TypeError):
                                            if not json_parts:
                                                add_text_part(reasoning_item.text)
                                    else:
                                        add_text_part(reasoning_item.text)
                    continue
                
                # Prioritize JSON content when using structured outputs
                if getattr(content_part, "json", None) is not None:
                    add_json_part(content_part.json)
                
                # Only include text if not preferring JSON or if no JSON was found
                if not prefer_json or not json_parts:
                    if getattr(content_part, "text", None):
                        text_content = content_part.text
                        if prefer_json:
                            # Try to parse as JSON first
                            try:
                                json.loads(text_content)
                                add_json_part(text_content)
                            except (json.JSONDecodeError, TypeError):
                                if not json_parts:
                                    add_text_part(text_content)
                        else:
                            add_text_part(text_content)
                    if getattr(content_part, "output_text", None):
                        output_text = content_part.output_text
                        if prefer_json:
                            try:
                                json.loads(output_text)
                                add_json_part(output_text)
                            except (json.JSONDecodeError, TypeError):
                                if not json_parts:
                                    add_text_part(output_text)
                        else:
                            add_text_part(output_text)

            if getattr(output_entry, "text", None):
                if not prefer_json or not json_parts:
                    add_text_part(output_entry.text)
            if getattr(output_entry, "json", None) is not None:
                add_json_part(output_entry.json)

        # When preferring JSON (structured outputs), return only JSON parts
        if prefer_json and json_parts:
            # If multiple JSON parts, try to combine them or return the first valid one
            # For strict JSON schema mode, there should only be one JSON object
            if len(json_parts) == 1:
                return json_parts[0]
            else:
                # Multiple JSON parts - try to parse each and return the first valid array/object
                for json_str in json_parts:
                    try:
                        parsed = json.loads(json_str)
                        if isinstance(parsed, (list, dict)):
                            return json_str
                    except json.JSONDecodeError:
                        continue
                # If all failed, return the first one anyway (will be handled by error handling)
                return json_parts[0]
        
        # Fallback to text parts or combined
        if text_parts:
            return "\n".join(text_parts)
        
        if json_parts:
            return json_parts[0]

        # As a last resort, attempt to serialize the whole response for debugging.
        # But first, check if there are any output entries we might have missed
        if not json_parts and not text_parts:
            # Try to find output in response structure
            for attr_name in ("output", "outputs", "content"):
                attr_value = getattr(response, attr_name, None)
                if attr_value:
                    if isinstance(attr_value, list):
                        for item in attr_value:
                            # Skip reasoning-only items
                            if getattr(item, "type", None) == "reasoning":
                                continue
                            # Check for actual content
                            if hasattr(item, "json") and item.json is not None:
                                add_json_part(item.json)
                            elif hasattr(item, "text") and item.text:
                                text_val = item.text
                                if prefer_json:
                                    try:
                                        json.loads(text_val)
                                        add_json_part(text_val)
                                    except (json.JSONDecodeError, TypeError):
                                        if not json_parts:
                                            add_text_part(text_val)
                                else:
                                    add_text_part(text_val)
                    elif isinstance(attr_value, dict):
                        # Check dict for content
                        if "json" in attr_value and attr_value["json"] is not None:
                            add_json_part(attr_value["json"])
                        elif "text" in attr_value and attr_value["text"]:
                            text_val = attr_value["text"]
                            if prefer_json:
                                try:
                                    json.loads(text_val)
                                    add_json_part(text_val)
                                except (json.JSONDecodeError, TypeError):
                                    if not json_parts:
                                        add_text_part(text_val)
                            else:
                                add_text_part(text_val)
            
            # If we found something, return it
            if prefer_json and json_parts:
                return json_parts[0] if len(json_parts) == 1 else json_parts[0]
            if text_parts:
                return "\n".join(text_parts)
        
        # Final fallback: serialize the whole response
        for attr_name in ("model_dump", "to_dict", "dict"):
            attr = getattr(response, attr_name, None)
            if callable(attr):
                try:
                    data = attr()
                    # Don't serialize reasoning-only responses
                    if isinstance(data, dict) and data.get("type") == "reasoning" and not data.get("content"):
                        continue
                except TypeError:
                    try:
                        data = attr({})
                        if isinstance(data, dict) and data.get("type") == "reasoning" and not data.get("content"):
                            continue
                    except Exception:
                        continue
                try:
                    serialized = json.dumps(data)
                    # Don't return reasoning-only metadata
                    if '"type":"reasoning"' in serialized and '"content":null' in serialized:
                        continue
                    return serialized
                except Exception:
                    continue
        if hasattr(response, "__dict__"):
            try:
                serialized = json.dumps(response.__dict__)
                # Don't return reasoning-only metadata
                if '"type":"reasoning"' in serialized and '"content":null' in serialized:
                    return ""
                return serialized
            except Exception:
                pass
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

    # Debug: Check response type
    print(f"\n=== DEBUG: Response type ===")
    print(f"Response type: {type(response)}")
    print(f"Has __iter__: {hasattr(response, '__iter__')}")
    print(f"Has status: {hasattr(response, 'status')}")
    print(f"Has __next__: {hasattr(response, '__next__')}")
    if hasattr(response, 'status'):
        print(f"Status: {getattr(response, 'status', None)}")
    
    # Handle streaming response
    # Check if response is a stream - streams are iterable but don't have status immediately
    is_stream = False
    try:
        # Check if it's iterable (but not a string/bytes/dict)
        if hasattr(response, '__iter__') and not isinstance(response, (str, bytes, dict)):
            # Additional check: if it has a status attribute, it's likely not a stream
            # Streams return events, not a single response object
            if not hasattr(response, 'status'):
                is_stream = True
                print("DEBUG: Detected as stream (no status attribute)")
            # Also check if it's a generator/iterator type
            elif hasattr(response, '__next__'):
                is_stream = True
                print("DEBUG: Detected as stream (has __next__)")
        else:
            print("DEBUG: Not detected as stream")
    except Exception as e:
        print(f"DEBUG: Exception checking stream: {e}")
        pass
    
    print(f"DEBUG: is_stream = {is_stream}")
    
    streamed_text = ""  # capture streamed output_text for final parsing
    if is_stream:
        # This is a stream - iterate through events
        accumulated_reasoning = ""  # used for UI updates (reasoning/output text)
        accumulated_output_text = ""  # used for final JSON parsing when streaming text
        final_response = None
        event_count = 0
        reasoning_items = {}  # Track reasoning items by ID to accumulate content
        
        print("DEBUG: Starting to process stream...")
        try:
            # Process stream with timeout handling
            import signal
            import sys
            
            for event in response:
                # Add a small delay to allow stream to process (but this shouldn't be necessary)
                # The stream should yield events as they come
                event_count += 1
                print(f"\nDEBUG: Processing event #{event_count}")
                print(f"DEBUG: Event type: {type(event)}")
                print(f"DEBUG: Event attributes: {dir(event)}")
                
                # Extract reasoning text from stream events
                # Stream events can have different structures - check common patterns
                event_type = getattr(event, 'type', None) or type(event).__name__
                print(f"DEBUG: Event type value: {event_type}")
                
                # Handle different event types
                output_item = None
                content_delta = None
                
                # Check for content delta events (these contain the actual reasoning text chunks)
                # Look for events that have delta, text, or content attributes
                # These come after the reasoning item is created
                delta_text = None
                if hasattr(event, 'delta'):
                    delta_obj = getattr(event, 'delta', None)
                    print(f"DEBUG: Event has delta attribute: {type(delta_obj)}")
                    if delta_obj:
                        if hasattr(delta_obj, 'text'):
                            delta_text = getattr(delta_obj, 'text', None)
                        elif isinstance(delta_obj, str):
                            delta_text = delta_obj
                        elif isinstance(delta_obj, dict):
                            delta_text = delta_obj.get('text') or delta_obj.get('content')
                
                # Also check event directly for text/content (some events might have it directly)
                if not delta_text:
                    if hasattr(event, 'text'):
                        delta_text = getattr(event, 'text', None)
                        print(f"DEBUG: Event has text attribute: {delta_text[:50] if delta_text else None}...")
                    elif hasattr(event, 'content'):
                        content_val = getattr(event, 'content', None)
                        if isinstance(content_val, str):
                            delta_text = content_val
                        print(f"DEBUG: Event has content attribute: {delta_text[:50] if delta_text else None}...")
                
                if delta_text:
                    print(f"DEBUG: Found delta text! Length: {len(delta_text)}")
                    # Track by item_id if available
                    item_id = None
                    if hasattr(event, 'item_id'):
                        item_id = getattr(event, 'item_id', None)
                        print(f"DEBUG: Delta item ID: {item_id}")
                        if item_id and item_id not in reasoning_items:
                            reasoning_items[item_id] = ""
                    
                    # Accumulate the delta
                    accumulated_reasoning += delta_text  # UI
                    accumulated_output_text += delta_text  # final parse buffer
                    if item_id:
                        reasoning_items[item_id] += delta_text
                    
                    print(f"DEBUG: Accumulated reasoning from delta: {len(delta_text)} chars, total: {len(accumulated_reasoning)}")
                    if progress_callback:
                        try:
                            progress_callback(delta_text)
                            print("DEBUG: Called progress_callback with delta successfully")
                        except Exception as e:
                            print(f"DEBUG: Error in callback: {e}")
                            import traceback
                            traceback.print_exc()
                    continue  # Skip to next event - we've handled this delta
                
                # ResponseOutputItemAddedEvent has 'item' attribute
                if hasattr(event, 'item'):
                    print(f"DEBUG: Event has item attribute")
                    output_item = getattr(event, 'item', None)
                # Some events might have 'output' directly
                elif hasattr(event, 'output'):
                    print(f"DEBUG: Event has output attribute")
                    output_list = getattr(event, 'output', [])
                    if isinstance(output_list, list) and len(output_list) > 0:
                        output_item = output_list[0]  # Take first item
                    elif output_list:
                        output_item = output_list
                # ResponseInProgressEvent might have 'response' with output that gets updated incrementally
                elif hasattr(event, 'response'):
                    print(f"DEBUG: Event has response attribute")
                    response_obj = getattr(event, 'response', None)
                    if response_obj:
                        output_list = getattr(response_obj, 'output', None)
                        print(f"DEBUG: Response output list: {type(output_list)}, length: {len(output_list) if isinstance(output_list, list) else 'N/A'}")
                        if isinstance(output_list, list) and len(output_list) > 0:
                            # Check all output items for reasoning with content
                            for idx, item in enumerate(output_list):
                                item_type = getattr(item, 'type', None)
                                item_id = getattr(item, 'id', None)
                                
                                # Try multiple ways to get content
                                item_content = getattr(item, 'content', None)
                                item_text = getattr(item, 'text', None)
                                item_output_text = getattr(item, 'output_text', None)
                                
                                # Get the actual content value
                                actual_content = item_content or item_text or item_output_text
                                
                                print(f"DEBUG: Output item #{idx}: type={item_type}, id={item_id}, has_content={actual_content is not None}")
                                
                                if item_type == 'reasoning':
                                    # Initialize tracking if needed
                                    if item_id and item_id not in reasoning_items:
                                        reasoning_items[item_id] = ""
                                    
                                    if actual_content:
                                        # This reasoning item has content! Extract it
                                        current_content = reasoning_items.get(item_id, "")
                                        
                                        # Check if this is new content
                                        if isinstance(actual_content, str):
                                            if actual_content != current_content:
                                                # Extract incremental text
                                                if actual_content.startswith(current_content):
                                                    incremental = actual_content[len(current_content):]
                                                else:
                                                    incremental = actual_content
                                                
                                                reasoning_items[item_id] = actual_content
                                                accumulated_reasoning += incremental
                                                print(f"DEBUG: Found reasoning content in response! {len(incremental)} new chars")
                                                print(f"DEBUG: Content preview: {incremental[:100]}...")
                                                if progress_callback and incremental:
                                                    try:
                                                        progress_callback(incremental)
                                                        print("DEBUG: Called progress_callback successfully")
                                                    except Exception as e:
                                                        print(f"DEBUG: Error in callback: {e}")
                                                        import traceback
                                                        traceback.print_exc()
                            output_item = output_list[-1]  # Take last item for other processing
                
                # Skip processing output_item if we already handled a content delta
                if content_delta:
                    continue
                
                # Process the output item if we found one
                if output_item:
                    print(f"DEBUG: Processing output item")
                    print(f"DEBUG: Output item type: {type(output_item)}")
                    print(f"DEBUG: Output item attributes: {[attr for attr in dir(output_item) if not attr.startswith('_')]}")
                    
                    entry_type = getattr(output_item, 'type', None)
                    print(f"DEBUG: Output item type value: {entry_type}")
                    
                    if entry_type == 'reasoning':
                        print("DEBUG: Found reasoning entry!")
                        # Debug: Print all attributes and their values
                        print(f"DEBUG: Reasoning item dict: {output_item.model_dump() if hasattr(output_item, 'model_dump') else 'N/A'}")
                        
                        # Extract reasoning text - check multiple possible attributes
                        content = getattr(output_item, 'content', None)
                        text_content = getattr(output_item, 'text', None)
                        output_text = getattr(output_item, 'output_text', None)
                        encrypted_content = getattr(output_item, 'encrypted_content', None)
                        summary = getattr(output_item, 'summary', None)
                        
                        print(f"DEBUG: content: {content}")
                        print(f"DEBUG: text_content: {text_content}")
                        print(f"DEBUG: output_text: {output_text}")
                        print(f"DEBUG: encrypted_content: {encrypted_content}")
                        print(f"DEBUG: summary: {summary}")
                        
                        # Check if this is just a placeholder and we need to wait for content events
                        # Reasoning might come in separate content delta events
                        reasoning_chunk = None
                        if output_text:
                            reasoning_chunk = str(output_text)
                            print(f"DEBUG: Using output_text")
                        elif text_content:
                            reasoning_chunk = str(text_content)
                            print(f"DEBUG: Using text_content")
                        elif content:
                            if isinstance(content, str):
                                reasoning_chunk = content
                            elif isinstance(content, list):
                                reasoning_chunk = "".join(str(item) for item in content)
                            else:
                                reasoning_chunk = str(content)
                            print(f"DEBUG: Using content")
                        elif summary:
                            # Summary might contain reasoning text
                            if isinstance(summary, str):
                                reasoning_chunk = summary
                            elif isinstance(summary, list):
                                reasoning_chunk = " ".join(str(item) for item in summary)
                            print(f"DEBUG: Using summary")
                        
                        # Note: Reasoning content might come in separate delta events
                        # Track reasoning items by ID to accumulate content from multiple events
                        reasoning_id = getattr(output_item, 'id', None)
                        print(f"DEBUG: Reasoning item ID: {reasoning_id}")
                        
                        # Store reasoning item for tracking
                        if reasoning_id:
                            if reasoning_id not in reasoning_items:
                                reasoning_items[reasoning_id] = ""
                        
                        # Check for content delta events - these might come separately
                        # Look for response.content.delta events in the stream
                        print(f"DEBUG: Reasoning chunk length: {len(reasoning_chunk) if reasoning_chunk else 0}")
                        if reasoning_chunk:
                            # Extract only new text (incremental)
                            if reasoning_chunk.startswith(accumulated_reasoning):
                                incremental = reasoning_chunk[len(accumulated_reasoning):]
                                accumulated_reasoning = reasoning_chunk
                                print(f"DEBUG: Incremental update (prefix match): {len(incremental)} chars")
                            else:
                                incremental = reasoning_chunk
                                accumulated_reasoning += incremental
                                print(f"DEBUG: Incremental update (append): {len(incremental)} chars")
                            
                            if incremental and progress_callback:
                                print(f"DEBUG: Calling progress_callback with {len(incremental)} chars")
                                print(f"DEBUG: Chunk preview: {incremental[:100]}...")
                                try:
                                    progress_callback(incremental)
                                    print("DEBUG: Progress callback executed successfully")
                                except Exception as callback_error:
                                    print(f"DEBUG: Error in progress callback: {callback_error}")
                                    import traceback
                                    traceback.print_exc()
                            else:
                                print(f"DEBUG: Not calling callback - incremental: {bool(incremental)}, callback: {bool(progress_callback)}")
                    else:
                        print(f"DEBUG: Output item is not reasoning type, it's: {entry_type}")
                else:
                    print(f"DEBUG: No output item found in this event")
                
                # Check if this is the final event with the actual response
                # ResponseInProgressEvent has a 'response' attribute we can use
                if hasattr(event, 'response'):
                    response_obj = getattr(event, 'response', None)
                    if response_obj:
                        status = getattr(response_obj, 'status', None)
                        print(f"DEBUG: Response status: {status}")
                        if status == 'complete':
                            print("DEBUG: Found complete response, using as final response")
                            final_response = response_obj
                            # Don't break - might get more output items
                        elif status == 'incomplete':
                            print("DEBUG: Response is incomplete, continuing...")
                            final_response = response_obj
                            # Don't break - might get more events
                        else:
                            # Use the response object anyway for final extraction
                            if not final_response:
                                final_response = response_obj
                
                # Also check event type for completion
                if event_type == 'response.completed' or 'completed' in str(event_type).lower():
                    print("DEBUG: Found completion event")
                    if hasattr(event, 'response'):
                        final_response = getattr(event, 'response', None)
                    else:
                        final_response = event
                
                # Check if output item contains non-reasoning content (actual JSON output)
                if output_item and not final_response:
                    entry_type = getattr(output_item, 'type', None)
                    if entry_type and entry_type != 'reasoning':
                        print(f"DEBUG: Found non-reasoning output item: {entry_type}")
                        # This might be the final JSON output
                        if hasattr(event, 'response'):
                            final_response = getattr(event, 'response', None)
            
            print(f"DEBUG: Stream processing complete. Processed {event_count} events.")
            print(f"DEBUG: Final response: {final_response is not None}")
            print(f"DEBUG: Accumulated reasoning length: {len(accumulated_reasoning)}")
            print(f"DEBUG: Accumulated output_text length: {len(accumulated_output_text)}")
            
            # Use final response if we found one, otherwise try to use last event
            if final_response:
                print("DEBUG: Using final_response")
                response = final_response
            else:
                # Stream completed but no final response - this shouldn't happen with structured outputs
                # but we'll handle it gracefully
                print("DEBUG: Warning: Stream completed but no final response found")
                print("DEBUG: Trying to use response as-is (might be last event)")
            # capture streamed text for parsing
            streamed_text = accumulated_output_text
        except Exception as stream_error:
            print(f"DEBUG: Error processing stream: {stream_error}")
            import traceback
            traceback.print_exc()
            # Fall back - try to use response as-is
            # If we have streamed output text so far, use it anyway
            streamed_text = accumulated_output_text
    else:
        print("DEBUG: Not a stream, processing as regular response")

    # Check if response is incomplete (for non-streaming or final stream response)
    if response and hasattr(response, 'status'):
        response_status = getattr(response, "status", None)
        if response_status == "incomplete":
            incomplete_details = getattr(response, "incomplete_details", None)
            if incomplete_details:
                reason = getattr(incomplete_details, "reason", None)
                if reason == "max_output_tokens":
                    print(f"\n⚠️ WARNING: Response incomplete - hit max_output_tokens limit.")
                    print("The model used all tokens for reasoning and didn't generate the actual output.")
                    print("Consider increasing max_output_tokens or using a model with less reasoning overhead.\n")
                    return [{} for _ in meal_descriptions]

    try:
        print("\n=== OpenAI API Response ===")
        # Prefer streamed output text if available; otherwise extract from response
        if streamed_text and streamed_text.strip():
            message_content = streamed_text
            print("DEBUG: Using streamed output_text for parsing")
        else:
            # Always prefer JSON since we use structured outputs where possible; here we stream JSON text
            message_content = extract_message_content(response, prefer_json=True)
        message_content = message_content or ""
        print(message_content)
        print("===========================================\n")
        try:
            # Try to extract JSON even if there's extra text before/after
            message_content = message_content.strip()

            # Handle case where content might be a JSON string (double-encoded)
            # First, try to parse as JSON string
            try:
                parsed_string = json.loads(message_content)
                # If it parsed to a string, parse again to get the actual object
                if isinstance(parsed_string, str):
                    response_json = json.loads(parsed_string)
                else:
                    response_json = parsed_string
            except (json.JSONDecodeError, TypeError):
                # If that fails, try parsing directly as JSON object
                # If we have multiple concatenated JSON objects (common in streaming),
                # scan the string and decode sequential objects into an array.
                decoder = json.JSONDecoder()
                objs: List[Dict] = []
                idx = 0
                s = message_content
                while idx < len(s):
                    # Skip whitespace and any stray commas/newlines
                    while idx < len(s) and s[idx] in " \r\n\t,":
                        idx += 1
                    if idx >= len(s):
                        break
                    if s[idx] not in "[{":
                        # Not a JSON start; advance
                        idx += 1
                        continue
                    try:
                        obj, end = decoder.raw_decode(s, idx)
                        objs.append(obj)
                        idx = end
                        # Continue scanning for more objects
                    except json.JSONDecodeError:
                        idx += 1
                        continue
                if not objs:
                    # Last resort: try standard json.loads (may still fail)
                    response_json = json.loads(message_content)
                else:
                    # If the first element is an array, use it; otherwise, treat as array of objects
                    if isinstance(objs[0], list):
                        response_json = objs[0]
                    else:
                        response_json = objs

            substitutions_list = []

            # Responses API with JSON schema returns an object with "substitutions" array
            if isinstance(response_json, dict):
                # Check for "substitutions" key first (our schema structure)
                if "substitutions" in response_json:
                    possible_list = response_json.get("substitutions")
                    if isinstance(possible_list, list):
                        substitutions_list = [
                            item
                            for item in possible_list
                            if isinstance(item, dict)
                            and "original" in item
                            and "substitution" in item
                        ]
                # Also handle key "allergen_substitutions" (model produced)
                if not substitutions_list and "allergen_substitutions" in response_json:
                    possible_list = response_json.get("allergen_substitutions")
                    if isinstance(possible_list, list):
                        id_items = [
                            item for item in possible_list
                            if isinstance(item, dict) and "id" in item and "substitution" in item
                        ]
                        if id_items:
                            substitutions_list = [
                                {"original": ingredient_id_to_raw.get(item["id"], item["id"]), "substitution": item["substitution"]}
                                for item in id_items
                            ]
                else:
                    # Fallback: try other common keys
                    for key in ("meals", "items"):
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

            # Also handle direct array format (backward compatibility)
            elif isinstance(response_json, list):
                substitutions_list = [
                    item
                    for item in response_json
                    if isinstance(item, dict)
                    and "original" in item
                    and "substitution" in item
                ]

            # If nothing found yet, try to interpret as id-based items
            if not substitutions_list:
                # Case 1: top-level array of {id, substitution}
                if isinstance(response_json, list):
                    id_items = [
                        item for item in response_json
                        if isinstance(item, dict) and "id" in item and "substitution" in item
                    ]
                    if id_items:
                        substitutions_list = [
                            {"original": ingredient_id_to_raw.get(item["id"], item["id"]), "substitution": item["substitution"]}
                            for item in id_items
                        ]
                # Case 2: dict with items key array
                elif isinstance(response_json, dict):
                    for key in ("items", "results"):
                        possible = response_json.get(key)
                        if isinstance(possible, list):
                            id_items = [
                                item for item in possible
                                if isinstance(item, dict) and "id" in item and "substitution" in item
                            ]
                            if id_items:
                                substitutions_list = [
                                    {"original": ingredient_id_to_raw.get(item["id"], item["id"]), "substitution": item["substitution"]}
                                    for item in id_items
                                ]
                                break

            # If still empty and the text contains multiple JSON objects, scan for the first JSON array
            if not substitutions_list and "[" in message_content:
                decoder2 = json.JSONDecoder()
                for i in range(len(message_content)):
                    if message_content[i] == "[":
                        try:
                            alt_json, _ = decoder2.raw_decode(message_content[i:])
                            if isinstance(alt_json, list):
                                # Try id-based first
                                id_items = [
                                    item for item in alt_json
                                    if isinstance(item, dict) and "id" in item and "substitution" in item
                                ]
                                if id_items:
                                    substitutions_list = [
                                        {"original": ingredient_id_to_raw.get(item["id"], item["id"]), "substitution": item["substitution"]}
                                        for item in id_items
                                    ]
                                    break
                                # Fallback to original/substitution shape
                                os_items = [
                                    item for item in alt_json
                                    if isinstance(item, dict) and "original" in item and "substitution" in item
                                ]
                                if os_items:
                                    substitutions_list = os_items
                                    break
                        except Exception:
                            continue

            # Deduplicate by id if present (keep the last occurrence)
            seen_by_id: Dict[str, Dict[str, str]] = {}
            deduped: List[Dict[str, str]] = []
            for it in substitutions_list:
                it_id = it.get("id")
                if it_id:
                    seen_by_id[it_id] = it
                else:
                    deduped.append(it)
            if seen_by_id:
                deduped.extend(seen_by_id.values())
                substitutions_list = deduped

            print(f"FINAL SUBSTITUTIONS LIST: {substitutions_list}")

            # Build final mapping without post-sanitization; rely on prompt constraints
            formatted_substitutions_dict = {}
            for item in substitutions_list:
                if "original" in item and "substitution" in item:
                    sub_value = str(item["substitution"]).strip()
                    if sub_value:
                        formatted_substitutions_dict[item["original"]] = sub_value
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
