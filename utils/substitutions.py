from typing import Dict, List
from sqlalchemy.orm import Session

from utils.database import SubstitutionRule
from utils.openai_service import get_ai_substitutions


def get_substitution_rules(allergens: List[str], db: Session = None) -> Dict[str, str]:
    """Return substitution rules for selected allergens from the database."""
    custom_rules: Dict[str, str] = {}

    if db:
        for allergen in allergens:
            db_rules = db.query(SubstitutionRule).filter(
                SubstitutionRule.allergen == allergen
            ).all()

            for rule in db_rules:
                custom_rules[rule.original] = rule.replacement

    return custom_rules


def get_ai_substitutions_for_meal(
    meal_description: str,
    allergens: List[str],
    custom_rules: Dict[str, str],
) -> Dict[str, str]:
    """Get AI-powered substitutions for a specific meal description."""
    return get_ai_substitutions(meal_description, allergens, custom_rules)


def add_substitution_rule(allergen: str, original: str, replacement: str, db: Session):
    """Add a new substitution rule to the database."""
    rule = SubstitutionRule(
        allergen=allergen,
        original=original,
        replacement=replacement,
    )
    db.add(rule)
    db.commit()
    return rule


def delete_substitution_rule(rule_id: int, db: Session) -> bool:
    """Delete a substitution rule from the database."""
    rule = db.query(SubstitutionRule).filter(SubstitutionRule.id == rule_id).first()
    if rule:
        db.delete(rule)
        db.commit()
        return True
    return False
