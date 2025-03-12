
import os
import re
import traceback
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

# --------------------- Configuration -----------------------

# Load environment variables from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyD9kmbO735ZRG-Vnk-iegTodps0ASbQq7A")
persist_directory = "./db"  # Directory for ChromaDB to store vectors

# Check API key loaded
print(f"✅ Google API Key Loaded: {bool(GOOGLE_API_KEY)}")

# --------------------- App Initialization ------------------

app = FastAPI()

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------- Utility Functions -------------------

def sanitize_chat_id(chat_id: str) -> str:
    """
    Ensure chat_id follows Chroma collection name requirements:
    - 3-63 characters
    - Alphanumeric, underscores, hyphens
    """
    chat_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', chat_id)
    if len(chat_id) < 3:
        chat_id = f"chat_{chat_id.zfill(3)}"
    if len(chat_id) > 63:
        chat_id = chat_id[:63]
    return chat_id


from langchain_community.vectorstores import Chroma
from chromadb import PersistentClient

def process_pdf_with_chroma(content: bytes, chat_id: str):
    """Process PDF and store or append embeddings using Gemini and Chroma."""
    chat_id = sanitize_chat_id(chat_id)
    print(f"🚀 Starting processing PDF for chat_id: {chat_id}")

    # Save uploaded file temporarily
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    print(f"📄 Temporary PDF file created at: {tmp_path}")

    try:
        # Load documents from PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} documents from PDF.")
    finally:
        os.remove(tmp_path)  # Clean up
        print(f"🗑️ Temporary file removed.")

    if not documents:
        print("⚠️ No documents found in PDF. Skipping embedding process.")
        return

    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    print("✅ Initialized Google Generative AI Embeddings.")

    # Chroma client to manage collections
    chroma_client = PersistentClient(path=persist_directory)

    # list_collections now returns just names (list of strings)
    existing_collections = chroma_client.list_collections()
    print(f"📚 Existing collections: {existing_collections}")

    # Check if chat_id collection exists
    if chat_id in existing_collections:
        print(f"📚 Collection '{chat_id}' exists. Adding new documents.")
        # Load existing collection and add new docs
        vector_store = Chroma(
            collection_name=chat_id,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        vector_store.add_documents(documents)
        print(f"✅ Added {len(documents)} new documents to existing collection '{chat_id}'.")
    else:
        print(f"🆕 Collection '{chat_id}' not found. Creating new collection.")
        # Create new collection
        vector_store = Chroma.from_documents(
            documents,
            embeddings,
            collection_name=chat_id,
            persist_directory=persist_directory
        )
        print(f"✅ New Chroma collection '{chat_id}' created with {len(documents)} documents.")

    # No need to persist manually — handled automatically now
    print("💾 Chroma vector store operation completed successfully.")


def get_retriever(chat_id: str):
    """Retrieve Chroma collection for chat ID."""
    chat_id = sanitize_chat_id(chat_id)  # ✅ Sanitize chat ID
    print(f"🔍 Retrieving embeddings for chat_id: {chat_id}")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vector_store = Chroma(
        collection_name=chat_id,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    print("✅ Retriever initialized.")
    return vector_store.as_retriever()

def get_chat_response(question: str, retriever):
    """Generate answer using Gemini and retrieved context."""
    print(f"💬 Generating response for question: {question}")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1
    )
    print("✅ Gemini LLM initialized.")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    response = qa_chain.run(question)
    print(f"✅ Response generated: {response}")
    return response
# --------------------- API Routes --------------------------

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...), chat_id: str = Form(...)):
    """
    Upload PDF and process it into embeddings.
    """
    try:
        content = await file.read()
        process_pdf_with_chroma(content, chat_id)
        return {"message": f"✅ PDF processed and ready for chat. Chat ID: {sanitize_chat_id(chat_id)}"}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)
#create chat post request
@app.post("/chat/")
async def chat(chat_id: str = Form(...), message: str = Form(...)):
    """
    Chat using context from uploaded PDFs.
    """
    try:
        retriever = get_retriever(chat_id)
        answer = get_chat_response(message, retriever)
        return {"response": answer}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


# --------------------- Instructions -----------------------

# Run using:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
