from config import GOOGLE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_MODEL, LLM_MODEL, TOP_K, gemini_client, INTENT_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT, MAX_HISTORY
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY
)

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=GOOGLE_API_KEY
)

vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)

prompt = ChatPromptTemplate.from_template(f"""{RAG_SYSTEM_PROMPT}

Context: {{context}}
Conversation History:{{history}}
Question: {{question}}
Answer:""")

retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

chain = (
    {
        "context": RunnableLambda(lambda x: x["question"]) | retriever,
        "question": RunnableLambda(lambda x: x["question"]),
        "history": RunnableLambda(lambda x: x["history"])
    }
    | prompt
    | llm
    | StrOutputParser()
)

def classify_intent(question):
    response = gemini_client.models.generate_content(
        model=LLM_MODEL,
        config=genai.types.GenerateContentConfig(
            system_instruction=INTENT_SYSTEM_PROMPT
        ),
        contents=question
    )
    return response.text.strip().lower()

# Chat loop
def chat():
    history = []
    print("RAG Chatbot ready! Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() == "exit":
            break

        intent = classify_intent(question)
        print(f"[Intent: {intent}]")  

        if intent == "greeting":
            print("Assistant: Hey there! Ask me anything about Zippy Bites!\n")
        elif intent == "capability":
            print("Assistant: I can answer questions about Zippy Bites — menu, locations, offers, timings, and more!\n")
        elif intent == "out_of_scope":
            print("Assistant: I can only answer questions about Zippy Bites.\n")
        else:
            history_str = "\n".join([
                f"User: {h['question']}\nAssistant: {h['answer']}"
                for h in history
            ])
            response = chain.invoke({
                "question": question,
                "history": history_str
            })
            print(f"\nAssistant: {response}\n")

            history.append({"question": question, "answer": response})

            if len(history) > MAX_HISTORY:
                history = history[-MAX_HISTORY:]

if __name__ == "__main__":
    chat()