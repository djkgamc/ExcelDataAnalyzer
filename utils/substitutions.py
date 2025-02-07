from typing import Dict, List

def get_substitution_rules(allergens: List[str]) -> Dict:
    """
    Return substitution rules based on selected allergens
    """
    all_rules = {
        "Gluten": {
            "bread": "gluten-free bread",
            "pasta": "gluten-free pasta",
            "flour tortilla": "corn tortilla",
            "breadcrumbs": "gluten-free breadcrumbs",
            "wheat flour": "almond flour",
            "pizza crust": "gluten-free pizza crust",
            "rolls": "gluten-free rolls",
            "crackers": "gluten-free crackers"
        },
        "Dairy": {
            "milk": "almond milk",
            "cheese": "dairy-free cheese",
            "yogurt": "coconut yogurt",
            "butter": "plant-based butter",
            "cream": "coconut cream",
            "sour cream": "dairy-free sour cream"
        },
        "Nuts": {
            "peanut butter": "sunflower seed butter",
            "almond": "seeds",
            "cashew": "seeds",
            "walnut": "seeds",
            "pecan": "seeds"
        },
        "Eggs": {
            "egg": "egg substitute",
            "mayonnaise": "vegan mayonnaise",
            "egg noodles": "rice noodles"
        },
        "Soy": {
            "soy sauce": "coconut aminos",
            "tofu": "chickpeas",
            "edamame": "green peas",
            "soy milk": "oat milk"
        }
    }

    selected_rules = {}
    for allergen in allergens:
        if allergen in all_rules:
            selected_rules.update(all_rules[allergen])

    return selected_rules
