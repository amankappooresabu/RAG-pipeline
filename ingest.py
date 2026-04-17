import time
from config import gemini_client, index, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, DOCS_PATH
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader



def load_document(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    return " ".join([page.page_content for page in pages])
    
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)

def embed_text(text):
    result = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )
    return result.embeddings[0].values

def store_chunks(chunks):
    batch = []
    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i+1}/{len(chunks)}...")
        embedding = embed_text(chunk)
        batch.append({
            "id": f"chunk-{i}",
            "values": embedding,
            "metadata": {"text": chunk}
        })
        
        if len(batch) == 90:
            print("Upserting batch to Pinecone...")
            index.upsert(vectors=batch)
            batch = []
            print("Rate limit pause, waiting 62 seconds...")
            time.sleep(62)
    
    # Upsert any remaining chunks
    if batch:
        print("Upserting final batch...")
        index.upsert(vectors=batch)
    
    print("All chunks stored!")

if __name__ == "__main__":
    print("Loading document...")
    text = load_document(DOCS_PATH)
    
    print("Chunking...")
    chunks = chunk_text(text)
    print(f"Total chunks: {len(chunks)}")
    
    print("Embedding and storing...")
    store_chunks(chunks)
    
    print("Done! Vector DB is ready.")
