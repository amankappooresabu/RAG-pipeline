# RAG Chatbot

A conversational chatbot built with RAG (Retrieval-Augmented Generation) as a learning project to explore LangChain, Pinecone, and Google Gemini.

## How It Works

1. User question comes in
2. **Intent is classified** — greetings, capability questions, and out-of-scope queries are handled directly without hitting the vector store
3. **Relevant context is retrieved** from Pinecone using semantic search
4. **Gemini generates an answer** grounded in the retrieved context and conversation history

## Stack

- **LLM & Embeddings** — Google Gemini via `langchain-google-genai`
- **Vector Store** — Pinecone
- **Orchestration** — LangChain (LCEL chains)
- **Intent Classification** — Gemini via `google-genai` SDK

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in your GOOGLE_API_KEY and PINECONE credentials
```

## Configuration

All settings live in `config.py` — model names, index name, `TOP_K`, and `MAX_HISTORY`.

## Running

```bash
python main.py
```

Type your question and press Enter. Type `exit` to quit.