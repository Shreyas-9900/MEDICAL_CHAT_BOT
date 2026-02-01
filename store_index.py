from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Load PDF documents
extracted_data = load_pdf_file(data="Data/")
text_chunks = text_split(extracted_data)

# Load embeddings
embeddings = download_hugging_face_embeddings()

# ✅ Pinecone Client (real one)
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"

# ✅ Create index only if not exists
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

print("✅ Pinecone index ready!")

# ✅ Upload documents using LangChain VectorStore
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

print("✅ Documents successfully stored in Pinecone!")
