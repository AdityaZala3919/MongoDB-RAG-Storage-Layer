import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.2
)

# Text splitter
def chunk_text(text, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)


# Create FAISS index
def create_faiss_index(chunks, save_path):
    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )
    vectorstore.save_local(save_path)


# Retrieve chunks
def retrieve_chunks(query, faiss_path, top_k=5):
    vectorstore = FAISS.load_local(
        faiss_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vectorstore.similarity_search(query, k=top_k)
    return [doc.page_content for doc in docs]


# Generate answer
def generate_answer(context, question):
    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer the question strictly using the context below.

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content
