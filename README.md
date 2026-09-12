# 🔬 ResearchMind – Multi-Agent AI Research System

ResearchMind is a **LangGraph-orchestrated multi-agent AI research system** that searches the web, extracts relevant information, generates structured research reports, and improves them through an automated critic-writer feedback loop.

---

## 🚀 Features

- 🔎 Web research using **Tavily**
- 🌐 Webpage scraping using **BeautifulSoup**
- 🤖 LangChain tool-using agents
- 🧠 Groq LLM integration
- 🔄 LangGraph stateful workflow
- 🧐 Automated critic-based report evaluation
- ✍️ Automatic report revision based on critic feedback
- 🛡️ Retry and exponential backoff using **Tenacity**
- 🖥️ Interactive **Streamlit** interface
- 📜 Full pipeline execution trace
- 📥 Downloadable research reports

---

## 🏗️ Architecture

```text
                         Research Topic
                               │
                               ▼
                    ┌────────────────────┐
                    │    Search Agent    │
                    │      Tavily        │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │    Reader Agent    │
                    │   BeautifulSoup    │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │    Writer Chain    │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │    Critic Chain    │
                    └─────────┬──────────┘
                              │
                              ▼
                         Score >= 7?
                         /          \
                       YES           NO
                        │             │
                        ▼             ▼
                       END       Revise Report
                                      │
                                      └──────► Writer



LangGraph Workflow

Search
   │
   ▼
Read
   │
   ▼
Write
   │
   ▼
Critique
   │
   ├────────────── Score >= 7 ──────────────► END
   │
   └────────────── Score < 7
                         │
                         ▼
                    Revise Report
                         │
                         └──────────────► Write

Installation
1. Clone the repository
git clone https://github.com/AMIR-HASSAN001/ResearchMind-Multi-Agent-AI-Research-System.git
cd ResearchMind-Multi-Agent-AI-Research-System
2. Create a virtual environment
python -m venv .venv
3. Activate the virtual environment
macOS / Linux
source .venv/bin/activate
Windows
.venv\Scripts\activate
4. Install dependencies
pip install -r requirements.txt
🔑 Environment Variables
Create a .env file in the project root:
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
⚠️ Never commit your .env file or expose your API keys publicly.
▶️ Run the Application
Start the Streamlit application:
streamlit run app.py
If using the project's virtual environment directly:
.venv/bin/streamlit run app.py
The application will open in your browser.

