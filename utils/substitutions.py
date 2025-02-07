from typing import Dict, List
from utils.database import get_db, SubstitutionRule
from sqlalchemy.orm import Session

def get_substitution_rules(allergens: List[str], db: Session = None) -> Dict[str, str]:
    """
    Return substitution rules based on selected allergens from database,
    falling back to default rules if none found.
    Returns a flattened dictionary mapping original items to replacements.
    """
    rules = {}

    if db:
        for allergen in allergens:
            db_rules = db.query(SubstitutionRule).filter(
                SubstitutionRule.allergen == allergen
            ).all()

            if db_rules:
                for rule in db_rules:
                    rules[rule.original] = rule.replacement

    # Default rules as fallback
    default_rules = {
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

    # Add any missing rules from defaults
    for allergen in allergens:
        if allergen in default_rules:
            rules.update(default_rules[allergen])

    return rules

def add_substitution_rule(allergen: str, original: str, replacement: str, db: Session):
    """Add a new substitution rule to the database"""
    rule = SubstitutionRule(
        allergen=allergen,
        original=original,
        replacement=replacement
    )
    db.add(rule)
    db.commit()
    return rule