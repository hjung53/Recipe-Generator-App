
import streamlit as st
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass, asdict

# Imaging
from PIL import Image
import io
import base64

# Barcode (optional)
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    from pyzbar import pyzbar
    HAS_PYZBAR = True
except Exception:
    HAS_PYZBAR = False

# OpenAI (for image vision + fallback JSON forcing)
from openai import OpenAI

# LangChain / OpenAI wrappers
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser

# RAG engine
from rag_engine import generate_recipes_rag

# ---------------- Config ----------------
CONFIG_PATH = "config.json"
cfg = json.load(open(CONFIG_PATH, "r", encoding="utf-8")) if os.path.exists(CONFIG_PATH) else {}

APP_TITLE = cfg.get("app_config", {}).get("title", "AI Recipe Generator")
EMBED_MODEL = cfg.get("ai_config", {}).get("embedding_model", "text-embedding-3-small")
LLM_MODEL = cfg.get("ai_config", {}).get("model", "gpt-4o-mini")
VECTOR_STORE_PATH = cfg.get("storage", {}).get("vector_store_path", "recipe_vector_store")

# ---------------- Data classes ----------------
@dataclass
class Ingredient:
    name: str
    quantity: str = ""
    category: str = ""
    is_leftover: bool = False
    expiry_status: str = "fresh"  # fresh, soon, expired

@dataclass
class Recipe:
    name: str
    ingredients_needed: List[str]
    ingredients_available: List[str]
    missing_ingredients: List[str]
    instructions: List[str]
    cooking_time: str
    difficulty: str
    calories: str = ""
    suitable_for_leftovers: bool = False

# ---------------- Utilities ----------------
def _image_to_data_url(img_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _guess_mime(fname: Optional[str]) -> str:
    if not fname: return "image/png"
    name = fname.lower()
    if name.endswith(".jpg") or name.endswith(".jpeg"):
        return "image/jpeg"
    if name.endswith(".webp"):
        return "image/webp"
    if name.endswith(".png"):
        return "image/png"
    return "image/png"

# ---------------- Generator ----------------
class RecipeGenerator:
    def __init__(self, api_key: str):
        # Store API key
        self.api_key = api_key
        
        # LangChain chat model (JSON enforced)
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=cfg.get("ai_config", {}).get("temperature", 0.7),
            api_key=api_key,
            model_kwargs={ "response_format": {"type": "json_object"} }
        )
        # Embeddings for FAISS
        self.embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=api_key)
        # Native OpenAI client (vision)
        self.oa_client = OpenAI(api_key=api_key)

        self.vector_store = None
        self.load_or_create_vector_store()

    def load_or_create_vector_store(self):
        # Try to load FAISS; if not exists, build from local JSON (recipe_database.json) minimal seed
        try:
            from langchain_community.vectorstores import FAISS
        except Exception:
            self.vector_store = None
            return

        if os.path.isdir(VECTOR_STORE_PATH):
            try:
                self.vector_store = FAISS.load_local(
                    VECTOR_STORE_PATH, self.embeddings, allow_dangerous_deserialization=True
                )
                return
            except Exception:
                pass

        # Build minimal vector store from recipe_database.json if available
        docs = []
        if os.path.exists("recipe_database.json"):
            data = json.load(open("recipe_database.json", "r", encoding="utf-8"))
            for r in data.get("recipes", []):
                text = (
                    f"TITLE: {r.get('name','')}\n"
                    f"DESC: {r.get('description','')}\n"
                    f"INGREDIENTS: {', '.join(r.get('ingredients', []))}\n"
                    f"TAGS: {', '.join(r.get('tags', []))}\n"
                    f"LEFTOVER: {r.get('suitable_for_leftovers', False)}\n"
                )
                docs.append(Document(page_content=text, metadata={"title": r.get("name","")}))
        else:
            # Fallback seed
            sample_recipes = [
                "김치볶음밥: 김치, 밥, 계란, 파, 참기름",
                "피자 프리타타: 남은 피자, 계란, 우유, 치즈",
                "라면 볶음밥: 남은 라면, 밥, 계란, 김치"
            ]
            docs = [Document(page_content=t) for t in sample_recipes]

        from langchain_community.vectorstores import FAISS
        vs = FAISS.from_documents(docs, self.embeddings)
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        vs.save_local(VECTOR_STORE_PATH)
        self.vector_store = vs

    # ---------- Image analysis via Vision LLM ----------
    def analyze_image_for_ingredients(self, image_bytes: bytes, filename: Optional[str] = None) -> List[str]:
        # Use OpenAI chat completions with image
        mime = _guess_mime(filename or "")
        data_url = _image_to_data_url(image_bytes, mime)
        prompt_text = (
            "Analyze this image and identify what you see:\n\n"
            "IMPORTANT RULES:\n"
            "1. If you see a PREPARED/COOKED DISH (e.g., bulgogi, pizza, fried chicken, bibimbap, pasta), "
            "identify it as 'leftover [dish name]' (e.g., 'leftover bulgogi', 'leftover pizza').\n"
            "2. Only list individual ingredients if you see RAW/UNCOOKED ingredients (e.g., vegetables in a fridge, raw meat).\n"
            "3. Do NOT break down cooked dishes into their ingredients.\n\n"
            "Examples:\n"
            "- Image of bulgogi → ['leftover bulgogi'] (NOT beef, carrot, onion)\n"
            "- Image of pizza slice → ['leftover pizza'] (NOT cheese, dough, tomato)\n"
            "- Image of raw carrots and beef → ['carrot', 'beef']\n\n"
            "Return a JSON array of food items (use lowercase, Korean or English names are both fine)."
        )
        resp = self.oa_client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.2,
            response_format={ "type": "json_object" },
            messages=[
                {"role":"system","content":"You are an expert at identifying food items from images. You can distinguish between cooked dishes and raw ingredients."},
                {"role":"user","content":[
                    {"type":"text","text": prompt_text},
                    {"type":"image_url","image_url":{"url": data_url}}
                ]}
            ]
        )
        try:
            content = resp.choices[0].message.content
            data = json.loads(content)
            if isinstance(data, dict):
                for k in ["ingredients", "items", "list"]:
                    if k in data and isinstance(data[k], list):
                        return [str(x) for x in data[k]]
                flat = []
                for v in data.values():
                    if isinstance(v, list):
                        flat.extend(v)
                return [str(x) for x in flat]
            elif isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            pass
        return []

    # ---------- Barcode ----------
    def scan_barcode(self, image_data: bytes) -> Optional[str]:
        try:
            if HAS_PYZBAR and HAS_CV2:
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                codes = pyzbar.decode(image)
                if codes:
                    return codes[0].data.decode("utf-8", errors="ignore")
            if HAS_CV2 and hasattr(cv2, "barcode_BarcodeDetector"):
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                det = cv2.barcode_BarcodeDetector()
                ok, decoded_info, decoded_type, points = det.detectAndDecode(img)
                if ok and decoded_info:
                    return decoded_info[0]
        except Exception:
            return None
        return None

    # ---------- Categorize ----------
    def categorize_ingredients(self, ingredients: List[str]) -> List[Ingredient]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Categorize the following ingredients and identify if they are leftovers.
Categories: vegetables, proteins, dairy, grains, condiments, leftovers, others.
Return strictly as JSON array like:
[{"name":"ingredient","category":"category","is_leftover":true}]"""),
            ("user", f"Categorize these: {', '.join(ingredients)}")
        ])
        chain = prompt | self.llm | JsonOutputParser()
        try:
            result = chain.invoke({})
            return [Ingredient(**item) for item in result]
        except Exception:
            return [Ingredient(name=x, category=("leftovers" if "leftover" in x.lower() else "others"),
                               is_leftover=("leftover" in x.lower())) for x in ingredients]

    # ---------- Recommend ----------
    def recommend_recipes(self, ingredients: List[Ingredient], use_rag: bool = True) -> List[Dict]:
        ingredient_names = [ing.name for ing in ingredients]
        has_leftovers = any(ing.is_leftover for ing in ingredients)

        if use_rag:
            structured = {
                "ingredients": [{"name": n, "quantity": None, "state": None, "is_leftover": ("leftover" in n.lower())} for n in ingredient_names],
                "leftover_dishes": [n for n in ingredient_names if "leftover" in n.lower()],
                "notes": [], "language": "ko"
            }
            rag = generate_recipes_rag(structured, model=LLM_MODEL, k=5, api_key=self.api_key)
            ui = []
            for r in rag.get("recipes", []):
                uses = r.get("uses", [])
                uses_text = ", ".join(uses) if uses else "다양한 재료 활용"
                ui.append({
                    "name": r.get("title", "Recipe"),
                    "description": f"사용 재료: {uses_text}",
                    "uses_leftovers": any("leftover" in u for u in uses),
                    "difficulty": "medium",
                    "time": f"{r.get('time_minutes','~30')} min"
                })
            if ui:
                return ui

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a creative chef specializing in using leftover foods.
Return 3-5 recipes as JSON list with keys: name, description, uses_leftovers (bool), difficulty (easy/medium/hard), time."""),
            ("user", f"Available ingredients: {', '.join(ingredient_names)}\nHas leftovers: {has_leftovers}")
        ])
        chain = prompt | self.llm | JsonOutputParser()
        return chain.invoke({})

    # ---------- Details ----------
    def generate_recipe_details(self, recipe_name: str, available_ingredients: List[str]) -> Recipe:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate a detailed recipe with step-by-step instructions.
Return JSON with keys: name, ingredients_needed (list), instructions (list), cooking_time, difficulty, calories, suitable_for_leftovers (bool)."""),
            ("user", f"Recipe: {recipe_name}\nAvailable ingredients: {', '.join(available_ingredients)}")
        ])
        chain = prompt | self.llm | JsonOutputParser()
        data = chain.invoke({})

        need = data.get("ingredients_needed", [])
        avail = [a.lower() for a in available_ingredients]
        missing = [x for x in need if not any(a in x.lower() or x.lower() in a for a in avail)]

        return Recipe(
            name=data.get("name", recipe_name),
            ingredients_needed=need,
            ingredients_available=available_ingredients,
            missing_ingredients=missing,
            instructions=data.get("instructions", []),
            cooking_time=data.get("cooking_time", "~30 minutes"),
            difficulty=data.get("difficulty", "medium"),
            calories=data.get("calories", ""),
            suitable_for_leftovers=data.get("suitable_for_leftovers", True)
        )

    def suggest_substitutions(self, missing_ingredients: List[str], available_ingredients: List[str]) -> Dict[str, str]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Suggest substitutions for missing ingredients using available ones.
Return JSON object where keys are missing ingredients and values are suggested substitutions."""),
            ("user", f"Missing: {', '.join(missing_ingredients)}\nAvailable: {', '.join(available_ingredients)}")
        ])
        chain = prompt | self.llm | JsonOutputParser()
        return chain.invoke({})

    # ---------- Image Generation ----------
    def generate_cooking_step_image(self, recipe_name: str, step_number: int, step_description: str) -> Optional[str]:
        """Generate an image for a specific cooking step using DALL-E"""
        try:
            prompt = f"A clean, professional food photography style image showing the cooking process: {step_description}. Make it look appetizing and instructional, suitable for a recipe book. Focus on the cooking action described."
            
            response = self.oa_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            return response.data[0].url
        except Exception as e:
            st.warning(f"Could not generate image for step {step_number}: {str(e)}")
            return None

    # ---------- Video Generation (Sora) ----------
    def generate_cooking_video(self, recipe_name: str, instructions: List[str], duration: int = 30) -> Optional[str]:
        """
        Generate a cooking video using OpenAI Sora 2
        
        Args:
            recipe_name: Name of the recipe
            instructions: List of cooking instructions
            duration: Video duration in seconds (default: 30)
        """
        try:
            # Combine all instructions into a cohesive video prompt
            full_recipe = f"A {duration}-second professional cooking tutorial for {recipe_name}. "
            full_recipe += "Show the following steps with clear, appetizing visuals in a bright modern kitchen: "
            # Limit steps based on duration (roughly 5 seconds per step)
            max_steps = max(3, duration // 5)
            full_recipe += " Then, ".join([f"Step {i+1}: {step}" for i, step in enumerate(instructions[:max_steps])])
            full_recipe += " Use professional food photography lighting and smooth camera movements. Make it look like a cooking show."
            
            # Generate video using Sora 2 API
            st.info(f"🎬 Starting {duration}-second video generation with Sora 2... This may take several minutes.")
            
            # Try to pass duration parameter if supported by the API
            try:
                video = self.oa_client.videos.create(
                    model='sora-2',
                    prompt=full_recipe,
                    duration=duration  # Specify duration in seconds
                )
            except TypeError:
                # If duration parameter is not supported, rely on prompt
                video = self.oa_client.videos.create(
                    model='sora-2',
                    prompt=full_recipe
                )
            
            # Handle different possible response structures
            if hasattr(video, 'url'):
                return video.url
            elif hasattr(video, 'data') and len(video.data) > 0:
                if hasattr(video.data[0], 'url'):
                    return video.data[0].url
            elif hasattr(video, 'id'):
                # Video might be processing asynchronously
                st.info(f"Video is being generated (ID: {video.id}). Please check back in a few minutes.")
                return None
            else:
                st.warning(f"Unexpected response format: {video}")
                return None
            
        except AttributeError as e:
            st.error(f"⚠️ Sora API method not available in current OpenAI SDK version. Please update: pip install --upgrade openai")
            st.info(f"Error details: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Could not generate video: {str(e)}")
            st.info("💡 Tip: Make sure you have access to Sora 2 API. Check https://platform.openai.com/docs/guides/video-generation")
            return None

# ---------------- Streamlit App ----------------
st.set_page_config(page_title=APP_TITLE, page_icon="🥘", layout="wide")
st.title(APP_TITLE)
st.markdown("*Transform your leftovers into delicious meals!*")

# Initialize session state
if "ingredients" not in st.session_state:
    st.session_state.ingredients = []
if "recipes" not in st.session_state:
    st.session_state.recipes = []
if "recipe_history" not in st.session_state:
    st.session_state.recipe_history = []

with st.sidebar:
    st.header("⚙️ Settings")
    
    # Try to get API key from secrets first, then allow manual input
    default_api_key = ""
    try:
        if "OPENAI_API_KEY" in st.secrets:
            default_api_key = st.secrets["OPENAI_API_KEY"]
    except Exception as e:
        pass
    
    # If API key exists in secrets, use it; otherwise ask for input
    if default_api_key:
        api_key = default_api_key
        st.success("✅ API key loaded from secrets")
    else:
        api_key = st.text_input("OpenAI API Key", type="password", 
                                help="Enter your OpenAI API key or save it in .streamlit/secrets.toml")
        if not api_key:
            st.warning("⚠️ Enter your OpenAI API key to continue")
            st.info("💡 Tip: Save your API key in `.streamlit/secrets.toml` to avoid entering it each time")
            st.stop()
        st.success("✅ API key set")
    
    use_rag = st.checkbox("Use FAISS retrieval (RAG)", value=True)
    generator = RecipeGenerator(api_key)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 Input Ingredients", "🍽️ Recipe Recommendations", "👨‍🍳 Recipe Details", "📊 History"])

with tab1:
    st.header("Add Your Ingredients")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📝 Text Input")
        text_input = st.text_area("Enter ingredients (one per line or comma-separated)",
                                  height=150,
                                  placeholder="e.g., leftover pizza, kimchi, rice, eggs...")
        if st.button("Add Text Ingredients", type="primary"):
            if text_input:
                new_ings = [x.strip() for x in text_input.replace("\n", ",").split(",") if x.strip()]
                categorized = generator.categorize_ingredients(new_ings)
                if "ingredients" not in st.session_state:
                    st.session_state.ingredients = []
                st.session_state.ingredients.extend(categorized)
                st.success(f"Added {len(new_ings)} ingredients!")
                st.rerun()

    with col2:
        st.subheader("📸 Image Input")
        up = st.file_uploader("Upload a photo of your fridge/leftovers",
                              type=["png", "jpg", "jpeg", "webp"])
        if up is not None:
            st.image(Image.open(up), caption="Uploaded", use_container_width=True)
            if st.button("Analyze Image", type="primary"):
                with st.spinner("Analyzing image (LLM Vision)..."):
                    items = generator.analyze_image_for_ingredients(up.getvalue(), filename=getattr(up, "name", ""))
                    if items:
                        categorized = generator.categorize_ingredients(items)
                        st.session_state.ingredients.extend(categorized)
                        st.success(f"Detected {len(items)} items")
                        st.rerun()
                    else:
                        st.error("No items detected. Try a clearer photo.")

    with col3:
        st.subheader("📊 Barcode Scanner")
        bc = st.file_uploader("Upload barcode image", type=["png","jpg","jpeg","webp"])
        if bc is not None:
            st.image(Image.open(bc), caption="Barcode Image", use_container_width=True)
            if st.button("Scan Barcode", type="primary"):
                code = generator.scan_barcode(bc.getvalue())
                if code:
                    st.success(f"Scanned: {code}")
                    categorized = generator.categorize_ingredients([f"product {code}"])
                    st.session_state.ingredients.extend(categorized)
                    st.rerun()
                else:
                    st.error("Could not scan barcode")

    if "ingredients" in st.session_state and st.session_state["ingredients"]:
        st.divider()
        st.subheader("📦 Current Ingredients")
        cats = {}
        for ing in st.session_state.ingredients:
            cats.setdefault(ing.category or "others", []).append(ing)
        cols = st.columns(max(1, len(cats)))
        for i, (cat, items) in enumerate(cats.items()):
            with cols[i]:
                st.markdown(f"**{cat.title()}**")
                for it in items:
                    prefix = "♻️ " if it.is_leftover else "• "
                    st.markdown(f"{prefix}{it.name}")
    else:
        st.info("Add ingredients via text, image, or barcode.")

with tab2:
    st.header("Recipe Recommendations")
    if "ingredients" not in st.session_state or not st.session_state.ingredients:
        st.info("👈 Please add ingredients first in the 'Input Ingredients' tab")
    else:
        if st.button("🎯 Get Recipe Recommendations", type="primary"):
            with st.spinner("Finding recipes..."):
                recs = generator.recommend_recipes(st.session_state.ingredients, use_rag=use_rag)
                st.session_state.recipes = recs

        if "recipes" in st.session_state and st.session_state.recipes:
            st.subheader("Recommended Recipes")
            for idx, recipe in enumerate(st.session_state.recipes):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"### {idx+1}. {recipe.get('name','(untitled)')}")
                    st.markdown(f"*{recipe.get('description','')}*")
                    tags = []
                    if recipe.get("uses_leftovers"):
                        tags.append("♻️ Uses Leftovers")
                    tags.append(f"⏱️ {recipe.get('time','~30 min')}")
                    tags.append(f"📊 {recipe.get('difficulty','medium').title()}")
                    st.markdown(" | ".join(tags))
                with col2:
                    if st.button("View Recipe", key=f"view_{idx}"):
                        st.session_state.selected_recipe = recipe.get("name")
                        st.rerun()
                st.divider()

with tab3:
    st.header("Recipe Details & Instructions")
    if st.session_state.get("selected_recipe"):
        with st.spinner(f"Generating recipe for {st.session_state['selected_recipe']}..."):
            available = [ing.name for ing in st.session_state.get("ingredients", [])]
            recipe = generator.generate_recipe_details(st.session_state["selected_recipe"], available)
            st.markdown(f"# {recipe.name}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cooking Time", recipe.cooking_time)
            c2.metric("Difficulty", recipe.difficulty.title())
            c3.metric("Calories", recipe.calories or "~500 kcal")
            if recipe.suitable_for_leftovers:
                c4.success("♻️ Leftovers Friendly")

            st.divider()
            left, right = st.columns(2)
            with left:
                st.subheader("✅ Available Ingredients")
                for ing in recipe.ingredients_available:
                    st.markdown(f"• {ing}")
            with right:
                st.subheader("🛒 Shopping List")
                if recipe.missing_ingredients:
                    for ing in recipe.missing_ingredients:
                        st.markdown(f"• {ing}")
                else:
                    st.success("You have all ingredients!")

            st.divider()
            st.subheader("👨‍🍳 Cooking Instructions")
            
            # Video/Image generation options
            col_option1, col_option2 = st.columns(2)
            
            with col_option1:
                generate_images = st.checkbox("🎨 Generate step-by-step images (DALL-E)", 
                                             help="Generate AI images for each cooking step (~$0.04 per image)")
            
            with col_option2:
                generate_video = st.checkbox("🎬 Generate cooking video (Sora 2)", 
                                            help="Generate a 1-minute cooking tutorial video using Sora 2 AI (may take several minutes and incur costs)",
                                            disabled=False)  # Sora 2 API is now available!
            
            # Video generation (Sora)
            if generate_video and not st.session_state.get(f"video_{recipe.name}"):
                # Duration selector
                video_duration = st.slider(
                    "Video duration (seconds)",
                    min_value=10,
                    max_value=60,
                    value=30,
                    step=5,
                    help="Select the duration for the generated video (10-60 seconds)"
                )
                
                if st.button("🎥 Generate Video", type="primary"):
                    with st.spinner(f"🎬 Generating {video_duration}-second cooking video... This may take a few minutes..."):
                        video_url = generator.generate_cooking_video(recipe.name, recipe.instructions, duration=video_duration)
                        if video_url:
                            st.session_state[f"video_{recipe.name}"] = video_url
                            st.rerun()
            
            if st.session_state.get(f"video_{recipe.name}"):
                st.video(st.session_state[f"video_{recipe.name}"])
                st.divider()
            
            # Step-by-step instructions with images
            for i, step in enumerate(recipe.instructions, 1):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Step {i}:** {step}")
                
                with col2:
                    if generate_images:
                        if f"step_image_{i}" not in st.session_state:
                            with st.spinner(f"Generating image for step {i}..."):
                                img_url = generator.generate_cooking_step_image(recipe.name, i, step)
                                st.session_state[f"step_image_{i}"] = img_url
                        
                        if st.session_state.get(f"step_image_{i}"):
                            st.image(st.session_state[f"step_image_{i}"], 
                                   caption=f"Step {i}",
                                   use_container_width=True)

            st.divider()
            action_col1, action_col2 = st.columns(2)
            with action_col1:
                if st.button("💾 Save Recipe", type="primary"):
                    if "recipe_history" not in st.session_state:
                        st.session_state.recipe_history = []
                    st.session_state.recipe_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "recipe": asdict(recipe)
                    })
                    st.success("Recipe saved to history!")
            
            with action_col2:
                if st.button("🔄 Clear Generated Media"):
                    # Clear all generated images and videos from session state
                    keys_to_remove = [k for k in st.session_state.keys() if k.startswith("step_image_") or k.startswith("video_")]
                    for key in keys_to_remove:
                        del st.session_state[key]
                    st.success("Cleared all generated images and videos")
                    st.rerun()
    else:
        st.info("👈 Select a recipe from the 'Recipe Recommendations' tab")

with tab4:
    st.header("Recipe History")
    for entry in reversed(st.session_state.get("recipe_history", [])):
        ts = entry["timestamp"]
        data = entry["recipe"]
        with st.expander(f"{data['name']} - {ts[:16].replace('T',' ')}"):
            st.markdown(f"**Cooking Time:** {data['cooking_time']}")
            st.markdown(f"**Difficulty:** {data['difficulty']}")
            st.markdown("**Instructions:**")
            for i, step in enumerate(data['instructions'], 1):
                st.markdown(f"{i}. {step}")
