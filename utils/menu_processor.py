import io
import re
import pandas as pd
from typing import Dict, List, Tuple
from utils.substitutions import get_ai_substitutions_for_meal

class MenuProcessor:
    def __init__(self, raw_content: str):
        self.raw_content = raw_content
        self.original_df = self.parse_menu()

    def parse_menu(self) -> pd.DataFrame:
        """Parse the raw menu text into a structured DataFrame"""
        # Read CSV content with no headers
        df = pd.read_csv(io.StringIO(self.raw_content), header=None)

        # Process each cell to extract meals
        processed_data = []

        for _, row in df.iterrows():
            for cell in row:
                if pd.notna(cell):
                    meals = self._parse_day_meals(cell)
                    processed_data.append(meals)

        # Convert to DataFrame
        result_df = pd.DataFrame(processed_data)
        return result_df

    def _parse_day_meals(self, day_text: str) -> Dict[str, str]:
        """Parse a single day's meals into breakfast, lunch, and snack"""
        meals = {
            'Breakfast': '',
            'Lunch': '',
            'Snack': ''
        }

        current_meal = None
        # Remove any quotes and split by newlines
        lines = day_text.replace('"', '').split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('B:'):
                current_meal = 'Breakfast'
                meals[current_meal] = line[2:].strip()
            elif line.startswith('L:'):
                current_meal = 'Lunch'
                meals[current_meal] = line[2:].strip()
            elif line.startswith('S:'):
                current_meal = 'Snack'
                meals[current_meal] = line[2:].strip()
            elif current_meal:
                meals[current_meal] += ' ' + line.strip()

        return meals

    def convert_menu(self, custom_rules: Dict[str, str], allergens: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Convert the menu using custom rules and AI-powered substitutions with batch processing"""
        from utils.openai_service import get_batch_ai_substitutions
        
        modified_df = self.original_df.copy()
        changes = []
        
        # Collect all meal descriptions to batch process
        all_meal_descriptions = []
        meal_indices = []  # Store (meal_type, idx) for each description
        
        for meal_type in ['Breakfast', 'Lunch', 'Snack']:
            if meal_type in modified_df.columns:
                for idx, description in enumerate(modified_df[meal_type]):
                    if pd.notna(description):  # Check if the meal description exists
                        all_meal_descriptions.append(description)
                        meal_indices.append((meal_type, idx))
        
        # Batch process all meals to get substitutions
        if all_meal_descriptions:
            # Get substitutions for all meals in one API call
            all_substitutions_list = get_batch_ai_substitutions(
                all_meal_descriptions,
                allergens,
                custom_rules
            )
            
            # Apply substitutions to each meal
            for i, (meal_type, idx) in enumerate(meal_indices):
                description = all_meal_descriptions[i]
                ai_substitutions = all_substitutions_list[i] if i < len(all_substitutions_list) else {}
                
                # Combine custom rules with AI substitutions, with AI taking precedence
                all_substitutions = {**custom_rules, **ai_substitutions}
                
                # Apply substitutions
                new_description = description
                for original, replacement in all_substitutions.items():
                    # Extract key terms for fuzzy matching (for items like "fish sticks" or partial words like "tartar sauc")
                    original_terms = original.lower().split()
                    description_lower = new_description.lower()
                    
                    # Check for partial matches of the original ingredient
                    # For multi-word ingredients (like "fish sticks"), we want to match if all key words appear near each other
                    # For single words, we want to match if a substantial part of the word is present
                    if len(original_terms) > 1:  # Multi-word term
                        # Check if all terms appear in the description near each other
                        all_terms_present = all(term in description_lower for term in original_terms)
                        if all_terms_present:
                            # Find the positions of each term
                            positions = [description_lower.find(term) for term in original_terms if term in description_lower]
                            # Check if terms are within a reasonable distance (e.g., 15 characters) of each other
                            if positions and max(positions) - min(positions) < 20:  # Adjust distance as needed
                                # Find the approximate match in the original text
                                # Extract the surrounding context
                                start_pos = max(0, min(positions) - 5)
                                end_pos = min(len(description_lower), max(positions) + len(original_terms[-1]) + 5)
                                context = description_lower[start_pos:end_pos]
                                
                                # Find words containing our terms in the context
                                for word in re.findall(r'\w+\s*\w*\s*\w*', context):
                                    if all(term in word for term in original_terms):
                                        # Match found - now substitute
                                        # Use a fuzzy pattern that will match the words and some surrounding context
                                        pattern = re.compile(re.escape(word), re.IGNORECASE)
                                        new_description = pattern.sub(replacement, new_description)
                                        changes.append(f"Changed '{word}' to '{replacement}' in {meal_type}")
                                        break
                    else:  # Single word term
                        # Special case for ingredients with typos or truncated words (like "tartar sauc" vs "tartar sauce")
                        # Look for words that start with the same characters (at least 70% of the word)
                        min_match_length = max(3, int(len(original_terms[0]) * 0.7))  # At least 3 chars or 70% of the word
                        
                        # Find all words in the description
                        words_in_description = re.findall(r'\b\w+\b', description_lower)
                        for word in words_in_description:
                            # Check if the word is a substantial match to the original term
                            original_term = original_terms[0]
                            if (original_term.startswith(word) and len(word) >= min_match_length) or \
                               (word.startswith(original_term) and len(original_term) >= min_match_length) or \
                               (len(original_term) > 4 and original_term[:-1] == word):  # Handle truncated words
                                # Replace the matched word
                                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                                new_description = pattern.sub(replacement, new_description)
                                changes.append(f"Changed '{word}' to '{replacement}' in {meal_type} (fuzzy match for '{original}')")
                        
                    # Standard exact match approach as a fallback
                    original_lower = original.lower()
                    if (original_lower in description_lower and  # Simple substring check
                        (original_lower == description_lower or  # Exact match
                         f" {original_lower} " in f" {description_lower} " or  # Surrounded by spaces
                         description_lower.startswith(f"{original_lower} ") or  # At start with space after
                         description_lower.endswith(f" {original_lower}") or  # At end with space before
                         original_lower in description_lower.split(', '))):  # In a comma-separated list
                        
                        # Perform a case-insensitive replacement
                        pattern = re.compile(re.escape(original), re.IGNORECASE)
                        new_description = pattern.sub(replacement, new_description)
                        changes.append(f"Changed '{original}' to '{replacement}' in {meal_type}")
                
                modified_df.at[idx, meal_type] = new_description
                
        return modified_df, changes

    def highlight_changes(self, original: str, modified: str) -> str:
        """Highlight the differences between original and modified text"""
        return modified