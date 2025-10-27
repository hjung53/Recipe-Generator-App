# 냉장고를 부탁해 - AI Recipe Generator (Streamlit + LangChain + FAISS)

남은 재료/배달음식을 입력(텍스트/이미지/바코드)하면 LLM이 **레시피**와 **쇼핑 리스트**를 제안합니다.
- 이미지 분석: Vision LLM (`gpt-4o-mini`)
- RAG: LangChain + FAISS (로컬 벡터스토어)
- UI: Streamlit



## 시연 영상


### **1) OpenAI API key 입력 → 재료 텍스트로 입력 → 메뉴 추천 → 시각적 레시피 생성 → 히스토리 저장**


https://github.com/user-attachments/assets/f0074fd1-6bbc-4b32-8c51-70574118164b



### **2) 재료 이미지로 입력 → 메뉴 추천 → 히스토리 저장**

![Image](https://github.com/user-attachments/assets/144a1eee-ff54-4f62-b7b2-5df5a319e40d)



## 1) 사전 준비

### 필수 설치
```bash
# (권장) 새 가상환경 생성 후
pip install -r requirements.txt
```

### OS별 바코드(ZBar) 라이브러리
- macOS: `brew install zbar`
- Ubuntu/Debian: `sudo apt-get install -y libzbar0`
- Windows: ZBar for Windows 설치(없으면 OpenCV fallback 사용)

### OpenAI 키 설정
```bash
# macOS/Linux
export OPENAI_API_KEY="sk-..."

# Windows PowerShell
setx OPENAI_API_KEY "sk-..."
```

## 2) 레시피 벡터스토어 생성 (최초 1회)

```bash
python build_recipe_index.py
```
- `recipe_database.json`의 레시피를 임베딩하여
- `config.json`의 `storage.vector_store_path`(기본: `recipe_vector_store`)에 저장합니다.

## 3) 애플리케이션 실행

```bash
streamlit run fridge_recipe_app.py
```
사이드바에 OpenAI API Key 입력 → (선택) **Use FAISS retrieval (RAG)** 체크 → 재료 입력 후 추천을 받아보세요.

## 4) RAG 단위 테스트 (선택)

```bash
python test_rag.py
```


## 폴더 구성
```
leftovers_app/
├── build_recipe_index.py
├── config.json
├── fridge_recipe_app.py
├── rag_engine.py
├── recipe_database.json
├── requirements.txt
├── test_rag.py
└── test_streamlit.py
```
# Recipe-Generator-App
