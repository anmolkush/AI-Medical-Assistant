from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# Extract data from the PDF
def load_pdf(data):
    try:
        loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading PDFs: {e}")
        return []

# Create text chunks
def text_split(extracted_data):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(extracted_data)
        return text_chunks
    except Exception as e:
        print(f"Error splitting text: {e}")
        return []

# Download embedding model
def download_hugging_face_embeddings():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return embeddings
    except Exception as e:
        print(f"Error downloading embeddings: {e}")
        return None
