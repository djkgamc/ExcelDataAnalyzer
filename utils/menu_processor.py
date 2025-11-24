import io
import re
from typing import Dict, List, Tuple, Set, Any

import pandas as pd


DAY_NAMES = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY"]
MEAL_LABELS = {"B": "Breakfast", "L": "Lunch", "S": "Snack"}


class MenuProcessor:
    def __init__(self, raw_content: bytes):
        self.raw_content = raw_content
        self.original_df = self.parse_menu()
        self.week_columns = self._find_week_columns()
        self.day_rows = self._find_day_rows()
        self.meal_cells = self._extract_meal_cells()
        self.substitution_map: Dict[Tuple[int, int], List[Tuple[str, str]]] = {}

    def parse_menu(self) -> pd.DataFrame:
        """Parse the raw Excel menu into a DataFrame while preserving layout."""
        try:
            df = pd.read_excel(io.BytesIO(self.raw_content), header=None)
        except Exception as exc:
            raise ValueError(
                "Unable to read the Excel file. Please confirm it matches the provided template."
            ) from exc
        return df

    def _find_week_columns(self) -> List[int]:
        """Locate columns that contain week headers (e.g., WEEK 1)."""
        week_cols = set()
        for col in self.original_df.columns:
            for value in self.original_df[col].dropna():
                if isinstance(value, str) and "WEEK" in value.upper():
                    week_cols.add(col)
        if not week_cols:
            raise ValueError("Could not locate week columns in the uploaded file.")
        return sorted(week_cols)

    def _find_day_rows(self) -> Dict[str, int]:
        """Locate rows that map to weekdays (Monday-Friday)."""
        day_rows: Dict[str, int] = {}
        for idx, row in self.original_df.iterrows():
            joined = " ".join(str(cell) for cell in row if pd.notna(cell))
            upper_row = joined.upper()
            for day in DAY_NAMES:
                if day in day_rows:
                    continue
                if day in upper_row or f"{day}S" in upper_row:
                    day_rows[day.capitalize()] = idx
        if len(day_rows) < len(DAY_NAMES):
            missing = [d.capitalize() for d in DAY_NAMES if d.capitalize() not in day_rows]
            raise ValueError(
                f"Could not find rows for all weekdays. Missing: {', '.join(missing)}."
            )
        return day_rows

    def _parse_meal_cell(self, cell_text: str) -> Dict[str, str]:
        """Extract B/L/S meal lines from a cell, even when multiple markers share a line."""
        meals: Dict[str, str] = {}

        # Search the full cell text for any "X: value" patterns instead of assuming
        # each marker starts a new line (the template sometimes has S: on the same
        # line as the lunch description).
        pattern = re.compile(r"([BLS])\s*:\s*([^\n]+)", flags=re.IGNORECASE)
        for match in pattern.finditer(str(cell_text)):
            meals[match.group(1).upper()] = match.group(2).strip()

        return meals

    def _extract_meal_cells(self) -> List[Dict[str, Any]]:
        """Collect all meal cells with their coordinates and parsed content."""
        cells: List[Dict[str, Any]] = []
        for week_index, col_idx in enumerate(self.week_columns, start=1):
            for day_name, row_idx in self.day_rows.items():
                cell_value = self.original_df.iloc[row_idx, col_idx]
                if pd.isna(cell_value) or not str(cell_value).strip():
                    raise ValueError(
                        f"Missing meal information for {day_name} in Week {week_index}."
                    )

                meal_parts = self._parse_meal_cell(str(cell_value))
                missing_meals = [label for label in MEAL_LABELS if label not in meal_parts]
                if missing_meals:
                    raise ValueError(
                        f"Could not find {', '.join(MEAL_LABELS[m] for m in missing_meals)} "
                        f"for {day_name} in Week {week_index}. Ensure each cell includes B:, L:, and S:."
                    )

                cells.append(
                    {
                        "week": week_index,
                        "day": day_name,
                        "row": row_idx,
                        "col": col_idx,
                        "text": str(cell_value),
                        "meal_parts": meal_parts,
                    }
                )
        return cells

    def _identify_meal_type(self, original_text: str, meal_parts: Dict[str, str]) -> str:
        """Best-effort mapping of a substitution back to Breakfast/Lunch/Snack."""
        for key, value in meal_parts.items():
            if original_text.lower() in value.lower():
                return MEAL_LABELS.get(key, "Meal")
        return "Meal"

    def convert_menu(
        self, custom_rules: Dict[str, str], allergens: List[str], progress_callback=None
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, List[Dict[str, str]]]]:
        """Convert the menu using custom rules and AI-powered substitutions.
        
        Args:
            custom_rules: Dictionary of custom substitution rules
            allergens: List of allergens to avoid
            progress_callback: Optional callback function(text: str) for streaming updates
        """
        modified_df = self.original_df.copy()
        self.substitution_map = {}

        all_changes: Set[str] = set()
        replaced_meals: List[Dict[str, str]] = []
        unreplaced_meals: List[Dict[str, str]] = []

        # Short-circuit if nothing to do
        if not allergens and not custom_rules:
            for cell in self.meal_cells:
                for meal_key, meal_text in cell["meal_parts"].items():
                    unreplaced_meals.append(
                        {
                            "week": cell["week"],
                            "day": cell["day"],
                            "meal_type": MEAL_LABELS.get(meal_key, "Meal"),
                            "text": meal_text,
                        }
                    )
            return modified_df, [], {"replaced": replaced_meals, "unreplaced": unreplaced_meals}

        cell_contents = [cell["text"] for cell in self.meal_cells]
        ai_substitutions: Dict[str, str] = {}

        # Only ask the AI for help when allergens are provided; custom rules alone
        # should not trigger API calls so we can support offline/local workflows
        # (and testing) without requiring an OpenAI key.
        if allergens and cell_contents:
            from utils.openai_service import get_batch_ai_substitutions
            all_substitutions_list = get_batch_ai_substitutions(
                cell_contents, allergens, custom_rules, progress_callback=progress_callback
            )
            if all_substitutions_list and len(all_substitutions_list) > 0:
                ai_substitutions = all_substitutions_list[0] or {}

        for cell in self.meal_cells:
            row_idx, col_idx = cell["row"], cell["col"]
            original_content = cell["text"]

            # Custom rules take precedence over AI suggestions
            all_substitutions = {**ai_substitutions, **custom_rules}

            new_content, cell_changes = self._apply_substitutions_to_cell(
                original_content, all_substitutions, row_idx, col_idx, cell["meal_parts"]
            )

            modified_df.iloc[row_idx, col_idx] = new_content
            all_changes.update(cell_changes)

            substitutions_made = self.get_substitutions_for_cell(row_idx, col_idx)
            if substitutions_made:
                replaced_meal_types: Set[str] = set()
                # Deduplicate substitutions: track unique (original, replacement) pairs per meal type
                seen_substitutions: Set[Tuple[str, str, str]] = set()
                for substitution_tuple in substitutions_made:
                    # Handle both old format (original, replacement) and new format (original, replacement, meal_type)
                    if len(substitution_tuple) == 3:
                        original, replacement, meal_type = substitution_tuple
                    else:
                        original, replacement = substitution_tuple
                        # Fallback to old identification method if meal_type not stored
                        meal_type = self._identify_meal_type(original, cell["meal_parts"])
                    
                    replaced_meal_types.add(meal_type)
                    # Only log if we haven't seen this exact substitution for this meal type in this cell
                    substitution_key = (meal_type, original.lower(), replacement.lower())
                    if substitution_key not in seen_substitutions:
                        seen_substitutions.add(substitution_key)
                        replaced_meals.append(
                            {
                                "week": cell["week"],
                                "day": cell["day"],
                                "meal_type": meal_type,
                                "original": original,
                                "replacement": replacement,
                            }
                        )
                for meal_key, meal_text in cell["meal_parts"].items():
                    meal_label = MEAL_LABELS.get(meal_key, "Meal")
                    if meal_label not in replaced_meal_types:
                        unreplaced_meals.append(
                            {
                                "week": cell["week"],
                                "day": cell["day"],
                                "meal_type": meal_label,
                                "text": meal_text,
                            }
                        )
            else:
                for meal_key, meal_text in cell["meal_parts"].items():
                    unreplaced_meals.append(
                        {
                            "week": cell["week"],
                            "day": cell["day"],
                            "meal_type": MEAL_LABELS.get(meal_key, "Meal"),
                            "text": meal_text,
                        }
                    )

        changes_list = sorted(list(all_changes))
        summary = {"replaced": replaced_meals, "unreplaced": unreplaced_meals}

        return modified_df, changes_list, summary

    def _apply_substitutions_to_cell(
        self, 
        content: str, 
        substitutions: Dict[str, str],
        row_idx: int,
        col_idx: int,
        meal_parts: Dict[str, str]
    ) -> Tuple[str, Set[str]]:
        """Apply substitutions to a single cell and track what changed"""
        new_content = content
        changes = set()
        cell_substitutions = []
        
        # Track all replacements with markers to avoid nested replacements
        replacements_to_apply = []
        
        # Build a map of character positions to meal types
        meal_type_map = {}
        for meal_key, meal_text in meal_parts.items():
            # Find the position of this meal marker (B:, L:, S:)
            marker_pattern = re.compile(rf"{meal_key}\s*:\s*", re.IGNORECASE)
            for marker_match in marker_pattern.finditer(content):
                start_pos = marker_match.end()
                # Find where this meal section ends (either next meal marker or end of content)
                next_marker_pattern = re.compile(rf"[BLS]\s*:\s*", re.IGNORECASE)
                next_match = next_marker_pattern.search(content, start_pos)
                end_pos = next_match.start() if next_match else len(content)
                # Mark all characters in this meal section
                for pos in range(start_pos, end_pos):
                    meal_type_map[pos] = MEAL_LABELS.get(meal_key, "Meal")
        
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
                start_pos = match.start()
                
                # Determine which meal type this occurrence belongs to
                meal_type = meal_type_map.get(start_pos, "Meal")
                # If exact position not found, try nearby positions
                if meal_type == "Meal":
                    for offset in range(max(0, start_pos - 10), min(len(content), start_pos + len(matched_text) + 10)):
                        if offset in meal_type_map:
                            meal_type = meal_type_map[offset]
                            break
                
                marker = f"__MARKER_{len(replacements_to_apply)}__"
                replacements_to_apply.append((matched_text, replacement, marker))
                changes.add(f"Changed '{matched_text}' to '{replacement}'")
                # Store with meal type: (original, replacement, meal_type)
                cell_substitutions.append((matched_text, replacement, meal_type))
        
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

    def get_substitutions_for_cell(self, row_idx: int, col_idx: int) -> List[Tuple[str, ...]]:
        """Get list of substitutions made in a specific cell.
        
        Returns list of tuples: (original, replacement) or (original, replacement, meal_type)
        """
        return self.substitution_map.get((row_idx, col_idx), [])
