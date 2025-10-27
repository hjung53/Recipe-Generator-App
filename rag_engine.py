
from __future__ import annotations
import os, json
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document

DEFAULT_INDEX_DIR = "recipe_vector_store"

def _load_config() -> Dict[str, Any]:
    for p in ("config.json", os.path.join(os.path.dirname(__file__), "config.json")):
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}

def _vectorstore_path_from_config() -> str:
    cfg = _load_config()
    try:
        return cfg["storage"]["vector_store_path"]
    except Exception:
        return DEFAULT_INDEX_DIR

def _embedding_model_from_config() -> str:
    cfg = _load_config()
    return cfg.get("ai_config", {}).get("embedding_model", "text-embedding-3-small")

INDEX_DIR = _vectorstore_path_from_config()

class Ingredient(BaseModel):
    name: str
    amount: str

class Substitution(BaseModel):
    missing: str
    suggestion: str

class Recipe(BaseModel):
    title: str
    uses: List[str] = Field(default_factory=list)
    servings: int = 2
    estimated_calories_per_serving: Optional[int] = None
    time_minutes: Optional[int] = None
    ingredients: List[Ingredient]
    substitutions: List[Substitution] = Field(default_factory=list)
    steps: List[str]
    safety_tips: List[str] = Field(default_factory=list)

class RecipesOut(BaseModel):
    recipes: List[Recipe]
    shopping_list: List[Ingredient] = Field(default_factory=list)

def _format_docs(docs: List[Document]) -> str:
    out = []
    for d in docs:
        title = d.metadata.get("title") or ""
        out.append(f"[{title}]\n{d.page_content}")
    return "\n\n---\n\n".join(out)

def _inventory_to_query(structured: Dict[str, Any]) -> str:
    names = [i.get("name", "") for i in structured.get("ingredients", [])]
    dishes = structured.get("leftover_dishes", [])
    lang = structured.get("language", "en")
    parts = []
    if names:
        parts.append("ingredients: " + ", ".join(names))
    if dishes:
        parts.append("leftovers: " + ", ".join(dishes))
    if lang == "ko":
        parts.append("prefer Korean recipes")
    return " | ".join(parts) if parts else "leftovers quick recipe"

def _ensure_vectorstore(api_key: Optional[str] = None) -> FAISS:
    embeddings = OpenAIEmbeddings(model=_embedding_model_from_config(), openai_api_key=api_key)
    if not os.path.isdir(INDEX_DIR):
        raise RuntimeError(
            f"FAISS index not found at {INDEX_DIR}. Build it first (python build_recipe_index.py)."
        )
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

def _build_chain(model: str = "gpt-4o-mini", api_key: Optional[str] = None):
    llm = ChatOpenAI(model=model, temperature=0.5, openai_api_key=api_key, model_kwargs={"response_format": {"type": "json_object"}})
    parser = JsonOutputParser(pydantic_object=RecipesOut)

    SYSTEM = """You are a leftover-focused recipe assistant.
Use the provided similar recipes (retrieved context) only as inspiration;
adapt them to use up the user's leftovers as much as possible.
Return JSON only that matches the given schema.
Follow food safety best practices (reheat leftovers ≥74°C; warn about risky items).
{format_instructions}
"""

    USER = """User inventory (normalized JSON):
{inventory}

Top similar recipes from the library:
{context}

Task:
- Propose 3 appetizing recipe ideas that maximize use of user's leftovers.
- In the "uses" field, list the specific ingredients from the user's inventory that each recipe uses.
- Keep steps concise and practical.
- If key items are missing, list substitutions and a short shopping list.
- Prefer Korean flavors if inputs or context suggest it.
"""

    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("user", USER)])
    chain = prompt | llm | parser
    return chain, parser

def generate_recipes_rag(inventory_struct: Dict[str, Any], *, model: str = "gpt-4o-mini", k: int = 5, api_key: Optional[str] = None) -> Dict[str, Any]:
    vs = _ensure_vectorstore(api_key=api_key)
    query = _inventory_to_query(inventory_struct)

    retriever = vs.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)  # LangChain v0.2+ retriever

    context_str = _format_docs(docs)
    chain, parser = _build_chain(model=model, api_key=api_key)
    result = chain.invoke({
        "inventory": json.dumps(inventory_struct, ensure_ascii=False),
        "context": context_str,
        "format_instructions": parser.get_format_instructions()
    })
    # Robust return handling: Pydantic object or plain dict/string
    try:
        if hasattr(result, "model_dump"):
            return result.model_dump()
        if hasattr(result, "dict"):
            return result.dict()
    except Exception:
        pass
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            return json.loads(result)
        except Exception:
            return {"recipes": [], "shopping_list": []}
    return {"recipes": [], "shopping_list": []}
