import pandas as pd
from typing import Dict, List, Tuple

class MenuProcessor:
    def __init__(self, df: pd.DataFrame):
        self.original_df = df.copy()
        self.validate_dataframe()

    def validate_dataframe(self):
        """Validate the input dataframe structure"""
        required_columns = ['Date', 'Meal', 'Description']
        missing_columns = [col for col in required_columns if col not in self.original_df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    def convert_menu(self, substitution_rules: Dict) -> Tuple[pd.DataFrame, List[str]]:
        """
        Convert the menu using provided substitution rules
        Returns modified dataframe and list of changes made
        """
        modified_df = self.original_df.copy()
        changes = []

        for idx, row in modified_df.iterrows():
            description = row['Description']
            for allergen, substitutions in substitution_rules.items():
                for original, replacement in substitutions.items():
                    if original.lower() in description.lower():
                        new_description = description.replace(original, replacement)
                        changes.append(f"Changed '{original}' to '{replacement}' in menu item: {description}")
                        modified_df.at[idx, 'Description'] = new_description

        return modified_df, changes

    def highlight_changes(self, original: str, modified: str) -> str:
        """Highlight the differences between original and modified text"""
        # This could be expanded to return HTML with highlighting
        return modified
