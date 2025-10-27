import json
from rag_engine import generate_recipes_rag

structured = {
    "ingredients": [
        {"name": "cooked rice", "quantity": None, "state": None, "is_leftover": True},
        {"name": "kimchi", "quantity": None, "state": None, "is_leftover": False},
        {"name": "eggs", "quantity": None, "state": None, "is_leftover": False},
        {"name": "scallions", "quantity": None, "state": None, "is_leftover": False}
    ],
    "leftover_dishes": ["fried chicken"],
    "notes": [],
    "language": "ko"
}

out = generate_recipes_rag(structured, model="gpt-4o-mini", k=5)
print(json.dumps(out, ensure_ascii=False, indent=2))
