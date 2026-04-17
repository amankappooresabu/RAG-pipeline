# config.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from google import genai

load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Clients
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Constants
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 10
DOCS_PATH = "docs/QSR.pdf"
MAX_HISTORY = 5

# Intent Prompts
INTENT_CATEGORIES = """
- greeting (hellos, hi, hey, how are you)
- capability (asking what you can do or help with)
- out_of_scope (anything unrelated to Zippy Bites restaurant)
- relevant (questions about Zippy Bites menu, locations, offers, timings, food items, branches)
"""

INTENT_SYSTEM_PROMPT = f"""You are an intent classifier.
Classify user messages into exactly one of these categories:
{INTENT_CATEGORIES}
Reply with just the category word, nothing else."""

# RAG System Prompt
RAG_SYSTEM_PROMPT = """You are a helpful assistant.
Answer the user's question based ONLY on the context provided.
If the answer is not in the context, say 'I don't know based on the provided document.'"""