import io
import unittest
from unittest.mock import patch

from utils.menu_processor import MenuProcessor


class MenuProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("Menu_Allergy_Sub.xlsm", "rb") as f:
            cls.sample_bytes = f.read()

    def test_extracts_all_meals_from_template(self):
        processor = MenuProcessor(self.sample_bytes)

        # 4 weeks * 5 weekdays = 20 cells
        self.assertEqual(len(processor.meal_cells), 20)

        # Each cell should have breakfast, lunch, and snack identified
        for cell in processor.meal_cells:
            self.assertIn("B", cell["meal_parts"])
            self.assertIn("L", cell["meal_parts"])
            self.assertIn("S", cell["meal_parts"])

    def test_custom_rules_do_not_trigger_ai_calls(self):
        processor = MenuProcessor(self.sample_bytes)
        custom_rules = {"Milk": "Soy milk"}

        with patch("utils.openai_service.get_batch_ai_substitutions") as mocked_ai:
            modified_df, changes, summary = processor.convert_menu(
                custom_rules, allergens=[]
            )

        mocked_ai.assert_not_called()

        # Ensure the substitution was applied somewhere in the data
        flattened_cells = "\n".join(
            str(cell) for cell in modified_df.values.flatten() if cell is not None
        )
        self.assertIn("Soy milk", flattened_cells)
        self.assertTrue(any("Soy milk" in change for change in changes))
        self.assertGreaterEqual(len(summary.get("replaced", [])), 1)

    def test_dairy_allergen_uses_custom_rules_and_ai_batch(self):
        processor = MenuProcessor(self.sample_bytes)
        custom_rules = {"Milk": "Soy milk"}
        ai_subs = {"Cheese": "Vegan cheese"}

        with patch(
            "utils.openai_service.get_batch_ai_substitutions",
            return_value=[ai_subs],
        ) as mocked_ai:
            modified_df, changes, summary = processor.convert_menu(
                custom_rules, allergens=["Dairy"]
            )

        # AI should be called once with all meal cells batched together
        mocked_ai.assert_called_once()
        call_args, call_allergens, call_rules = mocked_ai.call_args.args
        self.assertEqual(len(call_args), len(processor.meal_cells))
        self.assertEqual(call_allergens, ["Dairy"])
        self.assertEqual(call_rules, custom_rules)

        flattened_cells = "\n".join(
            str(cell) for cell in modified_df.values.flatten() if cell is not None
        )

        # Custom dairy rule is honored and AI handles remaining substitutions
        self.assertIn("Soy milk", flattened_cells)
        self.assertIn("Vegan cheese", flattened_cells)

        replacements = {item.get("replacement") for item in summary.get("replaced", [])}
        self.assertIn("Soy milk", replacements)
        self.assertIn("Vegan cheese", replacements)

        # Ensure the converted DataFrame can be written back to Excel
        buffer = io.BytesIO()
        modified_df.to_excel(buffer, index=False, header=False)
        self.assertGreater(buffer.tell(), 0)


if __name__ == "__main__":
    unittest.main()
