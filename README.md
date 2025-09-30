For App review check:https://www.linkedin.com/posts/zeyad-elsawi-0a0063284_summarising-the-second-half-of-my-internship-activity-7377827814198296576-JGjL?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAAEUQOrgB4AV9NKaTCISP1CP-uqMPn6OUd-0
🛒 2B Chat Bot

An AI-powered shopping assistant chatbot built with Streamlit + LangChain.
It answers user queries about products (e.g., “Show me 3 laptops between 20k–40k with 16GB RAM”) by retrieving from a CSV dataset and filtering dynamically.

🚀 Features

Natural language queries for products (budget, RAM, brand, etc.).

Uses LangChain Conversational Retrieval to recall chat history.

Integrates FAISS Vector Search for fast product retrieval.

Memory enabled: user can say “show me the first laptop again” and the bot remembers.

Two LLM options:

Google Generative AI (Gemini)

Ollama (LLaMA2, etc.)

Chat styled with custom Streamlit UI.

Automatic URL attachment from your dataset.

Downloadable chat history (.txt).

📂 Project Structure
.
├── Data.csv                # Your product dataset (update this with your own data)
├── app.py                  # Main Streamlit chatbot script
└── README.md               # Documentation

🛠️ Setup
1. Clone the repo
git clone https://github.com/your-username/2b-chatbot.git
cd 2b-chatbot

2. Install dependencies
pip install -r requirements.txt

3. Add your API keys

Open app.py

Replace this line:

os.environ["GOOGLE_API_KEY"] = "ADD YOUR KEY HERE"


Or uncomment Ollama if you want local LLMs:

# llm = Ollama(model="llama2")

4. Run the app
streamlit run app.py

📊 Data Format (CSV)

Your Data.csv should have at least these columns:

English Name	Price	RAM	Storage	Brand	url
Lenovo LOQ 15	32999	12GB	512GB SSD	Lenovo	https://2b.com/
...

✅ The chatbot automatically maps product names → URLs.
✅ Add more columns like GPU, Category, etc., for richer filtering.

⚠️ Important Note

The product information in this example is not guaranteed to be accurate.

You should use your own Data.csv file that contains data from your specific domain (e.g., laptops, mobiles, electronics, fashion).

Your CSV should include at least:

English Name → The product name

url or url_key → The product page link

Other useful fields (e.g., RAM, Storage, Brand, Price) depending on your use case.

This ensures the chatbot filters and recommends products dynamically based on your own dataset.

🧠 Memory

Uses ConversationBufferMemory to remember the last turns.

You can upgrade to ConversationBufferWindowMemory(k=10) if you want the bot to remember only the last 10 turns.

📥 Download Chat History

Users can download full chat logs as a .txt file.
