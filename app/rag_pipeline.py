import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
import google.generativeai as genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# ---- Gemini Embeddings Wrapper ----
class GeminiEmbeddings(Embeddings):
    model: str = "models/embedding-001"

    def embed_documents(self, texts):
        return [
            genai.embed_content(model=self.model, content=text)["embedding"]
            for text in texts
        ]

    def embed_query(self, text):
        return genai.embed_content(model=self.model, content=text)["embedding"]

# ---- Gemini LLM Wrapper ----
class GeminiLLM(LLM):
    model: str = "gemini-1.5-flash"

    def _call(self, prompt: str, stop=None):
        response = genai.GenerativeModel(self.model).generate_content(prompt)
        return response.text

    @property
    def _identifying_params(self):
        return {"model": self.model}

    @property
    def _llm_type(self):
        return "gemini"


def setup_vector_store(pdf_path):
    """
    Loads a local PDF, chunks it, and populates a Pinecone vector store.
    This function should be run once during application startup.
    """
    print("Starting vector store setup...")
    
    # Load PDF from local path
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
    except Exception as e:
        raise ValueError(f"Failed to load PDF from local path {pdf_path}: {str(e)}")

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # Create or reset Pinecone index
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if PINECONE_INDEX in existing_indexes:
        print(f"Deleting existing index: {PINECONE_INDEX}")
        pc.delete_index(PINECONE_INDEX)
        time.sleep(10)  # Wait for index deletion to complete
    
    print(f"Creating new index: {PINECONE_INDEX}")
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
    # Wait for index to initialize
    while not pc.describe_index(PINECONE_INDEX).status["ready"]:
        print("Waiting for index to be ready...")
        time.sleep(5)
    print("Index is ready.")

    # Store chunks in Pinecone using Gemini embeddings
    embeddings = GeminiEmbeddings()
    index = pc.Index(PINECONE_INDEX)
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )
    vectorstore.add_documents(chunks)
    
    print("Vector store setup complete.")
    return vectorstore

# ---- Main PDF Answering Function ----
def process_pdf_and_answer(vectorstore, questions):
    """
    Answers questions using a pre-populated vector store.
    """
    llm = GeminiLLM()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False
    )
    
    answers = []
    for q in questions:
        try:
            result = qa_chain.invoke({"query": q})
            answers.append(result["result"])
        except Exception as e:
            print(f"Error answering question: {e}")
            answers.append(f"Could not answer: {str(e)}")

    return answers