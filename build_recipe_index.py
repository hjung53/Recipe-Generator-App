import os, json
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

CONFIG_PATH = "config.json"
INDEX_DIR_DEFAULT = "recipe_vector_store"
DB_PATH = "recipe_database.json"

def load_config():
    if os.path.exists(CONFIG_PATH):
        return json.load(open(CONFIG_PATH, "r", encoding="utf-8"))
    return {}

def main():
    cfg = load_config()
    index_dir = cfg.get("storage", {}).get("vector_store_path", INDEX_DIR_DEFAULT)
    emb_model = cfg.get("ai_config", {}).get("embedding_model", "text-embedding-3-small")

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"{DB_PATH} not found")

    data = json.load(open(DB_PATH, "r", encoding="utf-8"))
    recipes = data.get("recipes", [])

    docs = []
    for r in recipes:
        text = (
            f"TITLE: {r.get('name','')}\n"
            f"DESC: {r.get('description','')}\n"
            f"INGREDIENTS: {', '.join(r.get('ingredients', []))}\n"
            f"TAGS: {', '.join(r.get('tags', []))}\n"
            f"LEFTOVER: {r.get('suitable_for_leftovers', False)}\n"
        )
        docs.append(Document(page_content=text, metadata={"title": r.get("name","")}))

    embeddings = OpenAIEmbeddings(model=emb_model)
    vs = FAISS.from_documents(docs, embedding=embeddings)

    os.makedirs(index_dir, exist_ok=True)
    vs.save_local(index_dir)
    print(f"Saved FAISS index to {index_dir}")

if __name__ == "__main__":
    main()
