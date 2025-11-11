import io
import re
import pandas as pd
from typing import Dict, List, Tuple, Set
from utils.substitutions import get_ai_substitutions_for_meal

class MenuProcessor:
    def __init__(self, raw_content: str):
        self.raw_content = raw_content
        self.original_df = self.parse_menu()
        self.substitution_map = {}  # Maps (row, col) -> list of (original_text, replacement_text)

    def parse_menu(self) -> pd.DataFrame:
        """Parse the raw menu text, preserving the original CSV structure"""
        df = pd.read_csv(io.StringIO(self.raw_content), header=None)
        return df

    def convert_menu(self, custom_rules: Dict[str, str], allergens: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Convert the menu using custom rules and AI-powered substitutions"""
        from utils.openai_service import get_batch_ai_substitutions
        
        modified_df = self.original_df.copy()
        self.substitution_map = {}
        
        # Track all changes
        all_changes = set()
        
        # Collect all cell contents for batch processing
        cell_contents = []
        cell_positions = []
        
        for row_idx in range(len(modified_df)):
            for col_idx in range(len(modified_df.columns)):
                cell_value = modified_df.iloc[row_idx, col_idx]
                if pd.notna(cell_value) and str(cell_value).strip():
                    cell_contents.append(str(cell_value))
                    cell_positions.append((row_idx, col_idx))
        
        # Get AI substitutions for all cells in batch
        # Note: The API returns ONE dictionary with all substitutions from the entire menu
        ai_substitutions = {}
        if cell_contents:
            all_substitutions_list = get_batch_ai_substitutions(
                cell_contents,
                allergens,
                custom_rules
            )
            # Extract the single dictionary of all substitutions
            if all_substitutions_list and len(all_substitutions_list) > 0:
                ai_substitutions = all_substitutions_list[0]
            
            # Process each cell with the same AI substitutions dictionary
            for i, (row_idx, col_idx) in enumerate(cell_positions):
                original_content = cell_contents[i]
                
                # Combine custom rules with AI substitutions
                all_substitutions = {**custom_rules, **ai_substitutions}
                
                # Apply substitutions and track changes
                new_content, cell_changes = self._apply_substitutions_to_cell(
                    original_content, 
                    all_substitutions,
                    row_idx,
                    col_idx
                )
                
                modified_df.iloc[row_idx, col_idx] = new_content
                all_changes.update(cell_changes)
        
        # Convert changes to list
        changes_list = sorted(list(all_changes))
        
        return modified_df, changes_list

    def _apply_substitutions_to_cell(
        self, 
        content: str, 
        substitutions: Dict[str, str],
        row_idx: int,
        col_idx: int
    ) -> Tuple[str, Set[str]]:
        """Apply substitutions to a single cell and track what changed"""
        new_content = content
        changes = set()
        cell_substitutions = []
        
        # Track all replacements with markers to avoid nested replacements
        replacements_to_apply = []
        
        for original, replacement in substitutions.items():
            if not original or not replacement:
                continue
                
            # Skip if already substituted
            if original.lower() == 'milk' and 'soy milk' in new_content.lower():
                continue
            
            # Find all occurrences (case-insensitive)
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            matches = list(pattern.finditer(new_content))
            
            for match in matches:
                matched_text = match.group(0)
                marker = f"__MARKER_{len(replacements_to_apply)}__"
                replacements_to_apply.append((matched_text, replacement, marker))
                changes.add(f"Changed '{matched_text}' to '{replacement}'")
                cell_substitutions.append((matched_text, replacement))
        
        # Apply markers
        temp_content = new_content
        for original, _, marker in replacements_to_apply:
            temp_content = temp_content.replace(original, marker, 1)
        
        # Replace markers with final text
        for _, replacement, marker in replacements_to_apply:
            temp_content = temp_content.replace(marker, replacement)
        
        new_content = temp_content
        
        # Store substitutions for this cell
        if cell_substitutions:
            self.substitution_map[(row_idx, col_idx)] = cell_substitutions
        
        return new_content, changes

    def get_substitutions_for_cell(self, row_idx: int, col_idx: int) -> List[Tuple[str, str]]:
        """Get list of substitutions made in a specific cell"""
        return self.substitution_map.get((row_idx, col_idx), [])
