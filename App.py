import os
import csv
import re

import streamlit as st
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import CSVLoader

# ======================
# STYLING
# ======================
st.set_page_config(page_title="2B Chat Bot", page_icon="🛒", layout="wide")

st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #ffffff 60%, #000000 40%);
            background-attachment: fixed;
        }
        .title {
            font-size: 42px;
            font-weight: bold;
            color: #ff6600;
            text-align: center;
            margin-bottom: 30px;
        }
        .chat-bubble-user {
            background-color: #000000;
            color: #ffffff;
            padding: 12px;
            border-radius: 15px;
            margin: 8px 0;
            text-align: right;
            max-width: 70%;
            margin-left: auto;
        }
        .chat-bubble-bot {
            background-color: #ffffff;
            color: #000000;
            padding: 12px;
            border-radius: 15px;
            margin: 8px 0;
            border: 1px solid #ff6600;
            max-width: 70%;
            margin-right: auto;
        }
        input {
            border-radius: 10px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🤖 2B Chat Bot</div>', unsafe_allow_html=True)








# ======================
# LLM
# ======================
os.environ["GOOGLE_API_KEY"] = "ADD  YOUR KEY HERE"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
# ======================
# LLM Option 2
# ======================
#llm = Ollama(model="llama2")
# ======================
# Embeddings + FAISS (CSV Loader, cached)
# ======================
@st.cache_resource
def load_vectorstore():
    csv_path = os.path.join(os.getcwd(), "Data.csv")
    loader = CSVLoader(file_path=csv_path, encoding='utf-8-sig')
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(data)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

with st.spinner("Loading product database..."):
    vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 15, "fetch_k": 40}
)

# Build a simple name -> URL map from the CSV for later augmentation
def load_name_to_url_map():
    mapping = {}
    try:
        csv_path = os.path.join(os.getcwd(), "Data.csv")
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("English Name") or "").strip()
                if not name:
                    continue
                # Prefer explicit URL column if present
                url = (row.get("url") or "").strip()
                url_key = (row.get("url_key") or "").strip()
                candidate = url or url_key
                if not candidate:
                    continue
                mapping[name.lower()] = candidate
    except Exception:
        pass
    return mapping

def _normalize_name(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

raw_name_to_url = load_name_to_url_map()
NAME_TO_URL = { _normalize_name(name): url for name, url in raw_name_to_url.items() }

def append_urls_to_answer(answer_text: str) -> str:
    if not NAME_TO_URL:
        return answer_text
    lines = answer_text.splitlines()
    new_lines = []
    for line in lines:
        # If the line already contains a URL, don't add another
        if re.search(r"https?://\S+", line):
            new_lines.append(line)
            continue
        norm_line = _normalize_name(line)
        matched_url = None
        for name_norm, url in NAME_TO_URL.items():
            if name_norm and name_norm in norm_line:
                matched_url = url
                break
        if matched_url:
            if "URL not provided" in line:
                line = line.replace("(URL not provided)", f"| URL: {matched_url}")
            elif "URL:" not in line and matched_url not in line:
                line = line.rstrip() + f" | URL: {matched_url}"
        new_lines.append(line)
    return "\n".join(new_lines)


# ======================
# Prompt Template
# ======================
template = """
You are a helpful shopping assistant for 2B products.
This is just an example, users can ask for any product:
The user will ask questions like "I need 3 laptops in budget 20k–40k with 16GB RAM and Lenovo only".
Use the retrieved laptop specs to filter dynamically by budget, RAM, storage, and brand,If the user didn't give all details provide options based on the ones he gave.
This is just an example, users can ask for any product.

You always use BOTH:
1. The retrieved product context
2. The full chat history

Rules:
- Users may refer to earlier results (e.g., "the first option", "that Lenovo one"). Always resolve this from chat history.
- Apply filters dynamically (budget, RAM, storage, brand, portability).
- Never invent products. If none match, say: "No items found within your filters."
- Return clean structured answers: Name, Specs, Price, URL.
- If asked for a summary, summarize the conversation from chat history.
Key Rules:
- For references to past recommendations (e.g., 'first laptop', 'that Lenovo one'): Quote exactly from Chat History. Do not add or alter details.
- Example Correct Recall: If history says "1. Lenovo Ideapad 3 Slim... 26599 EGP", respond with that exactly. Do not change to other models.
- Example Incorrect (Avoid): Do not say "Lenovo Legion 5" if not in history/context.
- Dynamic Filtering: Extract budget (e.g., 20-40k), RAM (>=4GB), brand from question. Apply filters only if relevant to the product category. For example, RAM and storage are for electronics like laptops and mobiles; ignore for air conditioners, wearables, or other categories unless specified in specs. If a filter like RAM is specified but not present in product specs, do not include those products.
- If missing and relevant to category, assume reasonable defaults (e.g., 8GB+ RAM for laptops/mobiles >20k) from context, but note it and ask for confirmation.
- Example: For "laptops 20-40k", filter by budget, suggest with 8-16GB RAM options, say "I assumed >=8GB RAM; specify if needed."
- For 'laptop or mobile', prioritize laptops if unspecified, but include mix if budget allows. Filter by category from context.
- Structure: List 1-3 items with Name, Specs (from context), Price (exact), URL (exact).
- If no matches: "No items found within your filters based on available data."
- Guide User: End with "Refine with more details (e.g., RAM, brand)?"

Chat History:
{chat_history}

User Question:
{question}

Retrieved Context:
{context}

Answer:"""

qa_prompt = PromptTemplate(
    input_variables=["chat_history", "question", "context"],
    template=template,
)



# ======================
# Memory + Chain
# ======================
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="question",
    output_key="answer"
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    return_source_documents=False,
    output_key="answer"
)

# ======================
# UI Interaction
# ======================
if st.button("Debug Retrieval"):
    test_query = "laptop 16GB RAM 20k-40k"
    docs = retriever.get_relevant_documents(test_query)
    for d in docs[:5]:
        st.write(d.page_content)
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_query = st.text_input("💬 Ask me about devices:")

if user_query:
    # Lightweight heuristic extraction to aid retrieval
    import re
    budget_match = re.search(r"(\d+)\s*-\s*(\d+)", user_query)
    ram_match = re.search(r"(\d+)\s*gb", user_query.lower())
    extracted = []
    if budget_match:
        extracted.append(f"budget {budget_match.group(1)}-{budget_match.group(2)}")
    if ram_match:
        extracted.append(f"ram {ram_match.group(1)}GB")
    enhanced_query = user_query + ("\n" + "; ".join(extracted) if extracted else "")

    st.session_state["chat_history"].append(("user", user_query))
    try:
        response = qa({"question": enhanced_query})
        final_answer = response["answer"]
        final_answer = append_urls_to_answer(final_answer)
        st.session_state["chat_history"].append(("bot", final_answer))
    except Exception as e:
        st.session_state["chat_history"].append(("bot", f"⚠️ Error: {e}"))

# ======================
# Display Chat History
# ======================
chat_text = ""
for role, message in st.session_state["chat_history"]:
    if role == "user":
        st.markdown(f'<div class="chat-bubble-user">{message}</div>', unsafe_allow_html=True)
        chat_text += f"User: {message}\n"
    else:
        st.markdown(f'<div class="chat-bubble-bot">{message}</div>', unsafe_allow_html=True)
        chat_text += f"Bot: {message}\n"

# ======================
# Download Chat History
# ======================
if chat_text:
    st.download_button(
        label="📥 Download Chat History (.txt)",
        data=chat_text,
        file_name="chat_history.txt",
        mime="text/plain"
    )
