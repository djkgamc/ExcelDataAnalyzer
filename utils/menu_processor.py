import io
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
                    if original.lower() in new_description.lower():
                        new_description = new_description.replace(original, replacement)
                        changes.append(f"Changed '{original}' to '{replacement}' in {meal_type}")
                
                modified_df.at[idx, meal_type] = new_description
                
        return modified_df, changes

    def highlight_changes(self, original: str, modified: str) -> str:
        """Highlight the differences between original and modified text"""
        return modified