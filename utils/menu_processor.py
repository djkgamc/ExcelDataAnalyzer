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
                
                # Debug output to verify what we're getting
                print(f"Meal type: {meal_type}, Meal: {description}")
                print(f"AI Substitutions: {ai_substitutions}")
                
                # Combine custom rules with AI substitutions, with AI taking precedence
                all_substitutions = {**custom_rules}
                
                # AI substitutions may be a dict with one item
                if isinstance(ai_substitutions, dict):
                    all_substitutions.update(ai_substitutions)
                
                # Apply substitutions
                new_description = description
                print(f"All substitutions: {all_substitutions}")
                
                for original, replacement in all_substitutions.items():
                    print(f"Checking for '{original}' in '{new_description}'")
                    # Simple direct string replacement
                    if original in new_description:
                        # Found exact match
                        new_description = new_description.replace(original, replacement)
                        changes.append(f"Changed '{original}' to '{replacement}' in {meal_type}")
                        print(f"Made substitution: '{original}' -> '{replacement}'")
                    # Case-insensitive search as fallback
                    elif original.lower() in new_description.lower():
                        # Find the actual case-preserved version in the text
                        pattern = re.compile(re.escape(original), re.IGNORECASE)
                        new_description = pattern.sub(replacement, new_description)
                        changes.append(f"Changed '{original}' to '{replacement}' in {meal_type} (case-insensitive)")
                        print(f"Made case-insensitive substitution: '{original}' -> '{replacement}'")
                    else:
                        print(f"No match found for '{original}'")
                        
                # Apply special case handling for common items containing allergens
                # Check first if both dairy and eggs are being removed
                if 'Egg Products' in allergens and 'Dairy' in allergens:
                    combo_allergen_items = {
                        '1/2 boiled egg': '1/2 scrambled tofu (egg-free, dairy-free alternative)',
                        'enriched corn muffin': 'vegan corn muffin (egg-free and dairy-free, made with applesauce)',
                        'enriched blueberry muffin': 'vegan blueberry muffin (egg-free and dairy-free, made with applesauce)',
                        'enriched banana bread': 'vegan banana bread (egg-free and dairy-free, made with applesauce)',
                        'enriched cinnamon raisin bread': 'vegan cinnamon raisin bread (egg-free and dairy-free)',
                        'pancake': 'vegan pancake (egg-free and dairy-free)',
                        'waffle': 'vegan waffle (egg-free and dairy-free)',
                        'muffin': 'vegan muffin (egg-free and dairy-free, made with applesauce)',
                        'buttermilk biscuit': 'vegan biscuit (egg-free and dairy-free)'
                    }
                    
                    for item, replacement in combo_allergen_items.items():
                        if item in new_description.lower() and not any(item in k.lower() for k in all_substitutions.keys()):
                            pattern = re.compile(re.escape(item), re.IGNORECASE)
                            match = pattern.search(new_description)
                            if match:
                                original_case = match.group(0)
                                new_description = new_description.replace(original_case, replacement)
                                changes.append(f"Changed '{original_case}' to '{replacement}' in {meal_type} (special case for egg+dairy)")
                                print(f"Made special case combo substitution: '{original_case}' -> '{replacement}'")
                
                elif 'Egg Products' in allergens:
                    egg_containing_items = {
                        '1/2 boiled egg': '1/2 scrambled tofu (egg-free alternative)',
                        'enriched corn muffin': 'egg-free corn muffin (made with applesauce)',
                        'enriched blueberry muffin': 'egg-free blueberry muffin (made with applesauce)',
                        'enriched banana bread': 'egg-free banana bread (made with applesauce)',
                        'enriched cinnamon raisin bread': 'egg-free cinnamon raisin bread',
                        'pancake': 'egg-free vegan pancake',
                        'waffle': 'egg-free vegan waffle',
                        'muffin': 'egg-free muffin (made with applesauce)'
                    }
                    
                    for item, replacement in egg_containing_items.items():
                        if item in new_description.lower() and not any(item in k.lower() for k in all_substitutions.keys()):
                            # Use a regular expression to match with case insensitivity but preserve case in output
                            pattern = re.compile(re.escape(item), re.IGNORECASE)
                            match = pattern.search(new_description)
                            if match:
                                original_case = match.group(0)
                                new_description = new_description.replace(original_case, replacement)
                                changes.append(f"Changed '{original_case}' to '{replacement}' in {meal_type} (special case for egg products)")
                                print(f"Made special case egg substitution: '{original_case}' -> '{replacement}'")
                
                if 'Fish' in allergens:
                    fish_containing_items = {
                        '3 fish sticks with tartar sauc': '3 plant-based fish-free sticks with vegan tartar sauce',
                        'tuna fish': 'chickpea "tuna" salad',
                        'fish sticks': 'plant-based fish-free sticks',
                        'tuna': 'chickpea "tuna" salad'
                    }
                    
                    for item, replacement in fish_containing_items.items():
                        if item in new_description.lower() and not any(item in k.lower() for k in all_substitutions.keys()):
                            # Use a regular expression to match with case insensitivity but preserve case in output
                            pattern = re.compile(re.escape(item), re.IGNORECASE)
                            match = pattern.search(new_description)
                            if match:
                                original_case = match.group(0)
                                new_description = new_description.replace(original_case, replacement)
                                changes.append(f"Changed '{original_case}' to '{replacement}' in {meal_type} (special case for fish)")
                                print(f"Made special case fish substitution: '{original_case}' -> '{replacement}'")
                
                if 'Dairy' in allergens:
                    dairy_containing_items = {
                        'milk': 'soy milk',
                        'cheese': 'dairy-free cheese alternative',
                        'yogurt': 'dairy-free yogurt',
                        'american cheese': 'dairy-free cheese alternative',
                        'cheddar': 'dairy-free cheddar alternative',
                        'mozzarella': 'dairy-free mozzarella alternative',
                        'ricotta': 'dairy-free ricotta alternative',
                        'butter': 'plant-based butter',
                        'cream cheese': 'dairy-free cream cheese',
                        'mac and cheese': 'dairy-free mac and cheese',
                        'macaroni & cheese': 'dairy-free macaroni & cheese',
                        'macaroni and cheese': 'dairy-free macaroni and cheese',
                        'ice cream': 'dairy-free ice cream',
                        # Add dairy substitutions for baked goods
                        'muffin': 'dairy-free muffin',
                        'corn muffin': 'dairy-free corn muffin',
                        'blueberry muffin': 'dairy-free blueberry muffin',
                        'pancake': 'dairy-free vegan pancake',
                        'waffle': 'dairy-free vegan waffle',
                        'buttermilk biscuit': 'dairy-free biscuit'
                    }
                    
                    for item, replacement in dairy_containing_items.items():
                        if item in new_description.lower() and not any(item in k.lower() for k in all_substitutions.keys()):
                            # Use a regular expression to match with case insensitivity but preserve case in output
                            pattern = re.compile(re.escape(item), re.IGNORECASE)
                            match = pattern.search(new_description)
                            if match:
                                original_case = match.group(0)
                                new_description = new_description.replace(original_case, replacement)
                                changes.append(f"Changed '{original_case}' to '{replacement}' in {meal_type} (special case for dairy)")
                                print(f"Made special case dairy substitution: '{original_case}' -> '{replacement}'")
                
                if 'Gluten' in allergens:
                    gluten_containing_items = {
                        'cereal': 'gluten-free cereal',
                        'bread': 'gluten-free bread',
                        'roll': 'gluten-free roll',
                        'pasta': 'gluten-free pasta',
                        'spaghetti': 'gluten-free spaghetti',
                        'macaroni': 'gluten-free macaroni',
                        'crackers': 'gluten-free crackers',
                        'bagel': 'gluten-free bagel',
                        'biscuit': 'gluten-free biscuit',
                        'muffin': 'gluten-free muffin',
                        'pretzel': 'gluten-free pretzel',
                        'waffle': 'gluten-free waffle',
                        'pancake': 'gluten-free pancake',
                        'bread': 'gluten-free bread',
                        'wgr': 'gluten-free'
                    }
                    
                    for item, replacement in gluten_containing_items.items():
                        if item in new_description.lower() and not any(item in k.lower() for k in all_substitutions.keys()):
                            # Use a regular expression to match with case insensitivity but preserve case in output
                            pattern = re.compile(r'\b' + re.escape(item) + r'\b', re.IGNORECASE)
                            match = pattern.search(new_description)
                            if match:
                                original_case = match.group(0)
                                new_description = new_description.replace(original_case, replacement)
                                changes.append(f"Changed '{original_case}' to '{replacement}' in {meal_type} (special case for gluten)")
                                print(f"Made special case gluten substitution: '{original_case}' -> '{replacement}'")
                
                modified_df.at[idx, meal_type] = new_description
                
        return modified_df, changes

    def highlight_changes(self, original: str, modified: str) -> str:
        """Highlight the differences between original and modified text"""
        return modified