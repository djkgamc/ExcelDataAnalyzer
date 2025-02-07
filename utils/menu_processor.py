import pandas as pd
from typing import Dict, List, Tuple

class MenuProcessor:
    def __init__(self, raw_content: str):
        self.raw_content = raw_content
        self.original_df = self.parse_menu()

    def parse_menu(self) -> pd.DataFrame:
        """Parse the raw menu text into a structured DataFrame"""
        # Split into rows (each row is a day)
        rows = self.raw_content.strip().split('\n')

        # Process each row into a list of meals
        processed_data = []
        current_day = []

        for row in rows:
            if row.strip():  # Skip empty rows
                meals = {}
                parts = row.split('\t')  # Split by tab to get different weeks

                for week_menu in parts:
                    day_meals = self._parse_day_meals(week_menu)
                    processed_data.append(day_meals)

        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        return df

    def _parse_day_meals(self, day_text: str) -> Dict[str, str]:
        """Parse a single day's meals into breakfast, lunch, and snack"""
        meals = {
            'Breakfast': '',
            'Lunch': '',
            'Snack': ''
        }

        current_meal = None
        parts = day_text.split('"')[1].split('\n') if '"' in day_text else day_text.split('\n')

        for part in parts:
            part = part.strip()
            if part.startswith('B:'):
                current_meal = 'Breakfast'
                meals[current_meal] = part[2:].strip()
            elif part.startswith('L:'):
                current_meal = 'Lunch'
                meals[current_meal] = part[2:].strip()
            elif part.startswith('S:'):
                current_meal = 'Snack'
                meals[current_meal] = part[2:].strip()
            elif current_meal and part:
                meals[current_meal] += ' ' + part.strip()

        return meals

    def convert_menu(self, substitution_rules: Dict) -> Tuple[pd.DataFrame, List[str]]:
        """Convert the menu using provided substitution rules"""
        modified_df = self.original_df.copy()
        changes = []

        for meal_type in ['Breakfast', 'Lunch', 'Snack']:
            if meal_type in modified_df.columns:
                for idx, description in enumerate(modified_df[meal_type]):
                    if pd.notna(description):  # Check if the meal description exists
                        new_description = description
                        for allergen, substitutions in substitution_rules.items():
                            for original, replacement in substitutions.items():
                                if original.lower() in new_description.lower():
                                    new_description = new_description.replace(original, replacement)
                                    changes.append(f"Changed '{original}' to '{replacement}' in {meal_type}")
                        modified_df.at[idx, meal_type] = new_description

        return modified_df, changes

    def highlight_changes(self, original: str, modified: str) -> str:
        """Highlight the differences between original and modified text"""
        # This could be expanded to return HTML with highlighting
        return modified