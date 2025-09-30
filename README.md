# 🛒 2B Chat Bot

An AI-powered shopping assistant chatbot built with Streamlit + LangChain. It answers user queries about products (e.g., “Show me 3 laptops between 20k–40k with 16GB RAM”) by retrieving from a CSV dataset and filtering dynamically.

- Preview/context: [LinkedIn post](https://www.linkedin.com/posts/zeyad-elsawi-0a0063284_summarising-the-second-half-of-my-internship-activity-7377827814198296576-JGjL?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAAEUQOrgB4AV9NKaTCISP1CP-uqMPn6OUd-0)

## Data Privacy (Important)

- The data included here (e.g., `Data.csv`, `Data.xlsx`) is private and provided only as placeholders.
- You must supply your own similarly structured data files to use this project.
- Ensure you have permission to use any data you load.

## Project Structure

## 🚀 Features

- Natural language queries for products (budget, RAM, brand, etc.)
- Conversational Retrieval (LangChain) to recall chat history
- FAISS Vector Search for fast product retrieval
- Memory enabled: “show me the first laptop again” works
- Two LLM options:
  - Google Generative AI (Gemini)
  - Ollama (e.g., LLaMA2, others)
- Chat styled with custom Streamlit UI
- Automatic URL attachment from your dataset
- Downloadable chat history (.txt)

## 🧩 Theory / Architecture

- Retrieval-Augmented Generation (RAG):
  - Documents are embedded and indexed with FAISS for vector similarity search.
  - A retriever selects relevant chunks for each user query.
  - The LLM generates answers grounded on retrieved context.
- Conversation Memory:
  - LangChain memory (e.g., ConversationBufferMemory) preserves recent turns.
  - Enables follow-ups and references like “the first one”.
- Orchestration:
  - LangChain Conversational Retrieval Chain wires together: memory + retriever + LLM.
  - Streamlit provides a lightweight chat UI and session state.

---

## 📂 Project Structure

```
.
├── Data.csv                 # Your product dataset (replace with your own data)
├── app.py                   # Main Streamlit chatbot script

└── README.md                # Documentation
```

---

## 🛠️ Setup

1) Clone the repo

```bash
git clone https://github.com/your-username/2b-chatbot.git
cd 2b-chatbot
```

2) Create and activate a virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt  # If present
# Or install common deps directly
pip install streamlit langchain faiss-cpu sentence-transformers pandas openpyxl \
            google-generativeai ollama transformers torch
```

4) Add your API keys (if using Gemini)

- Open `app.py`
- Replace this line:

```python
os.environ["GOOGLE_API_KEY"] = "ADD YOUR KEY HERE"
```

- Or use Ollama locally:

```python
from langchain_community.llms import Ollama
llm = Ollama(model="llama2")
```

5) Place your data

- Replace `Data.csv` with your own dataset (see format below)

---

## ▶️ Run the app

```bash
streamlit run app.py
```

Alternative (if an API server exists in your variant):

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

---

## 📊 Data Format (CSV)

Your `Data.csv` should have at least these columns:

- `English Name` → Product name
- `Price` → Numeric price
- `RAM` → e.g., `16GB`
- `Storage` → e.g., `512GB SSD`
- `Brand` → e.g., `Lenovo`
- `url` or `url_key` → Product page link

Example:

```csv
English Name,Price,RAM,Storage,Brand,url
Lenovo LOQ 15,32999,12GB,512GB SSD,Lenovo,https://2b.com/...
```

- ✅ The chatbot automatically maps product names to URLs.
- ✅ Add more columns like `GPU`, `Category`, etc., for richer filtering.

---

## ⚠️ Important Note

- The product information in any example data is not guaranteed to be accurate.
- You should use your own `Data.csv` that contains data from your specific domain (e.g., laptops, mobiles, electronics, fashion).
- Minimum recommended columns: `English Name`, `url` or `url_key`, plus relevant attributes like `RAM`, `Storage`, `Brand`, `Price`.

---

## 🧠 Memory

- Uses `ConversationBufferMemory` to remember previous turns.
- You can switch to `ConversationBufferWindowMemory(k=10)` to limit memory to the last 10 turns.

---

## 📥 Download Chat History

- Users can download the full chat log as a `.txt` file from the UI.

---

## 🔧 Model Notes

- If using local weights (e.g., TinyLlama) keep `models/tinyllama-1.1b-chat/` intact.
- To switch providers/models (Gemini, Ollama, HF models), adjust the LLM initialization in `app.py`.

---

## 📜 Disclaimer

This project is for educational purposes. Use at your own risk and comply with all data, privacy, and licensing requirements.
