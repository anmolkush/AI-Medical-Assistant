from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone  # Updated import
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore  # Updated VectorStore import
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")

print(f"Pinecone API Key: {PINECONE_API_KEY}")
print(f"Pinecone Environment: {PINECONE_API_ENV}")

# Load and process data
try:
    extracted_data = load_pdf("data/")
    if not extracted_data:
        raise ValueError("No documents extracted from the PDFs.")
except Exception as e:
    print(f"Error loading PDF data: {e}")
    extracted_data = []

# Split extracted text into chunks
try:
    text_chunks = text_split(extracted_data)
    print(len(text_chunks))
    # print(text_chunks)
    if not text_chunks:
        raise ValueError("No text chunks created.")
except Exception as e:
    print(f"Error splitting text: {e}")
    text_chunks = []

# Download Hugging Face embeddings
try:
    embeddings = download_hugging_face_embeddings()
    if embeddings is None:
        raise ValueError("Failed to download embeddings.")
except Exception as e:
    print(f"Error downloading embeddings: {e}")
    embeddings = None

# Proceed only if embeddings and data are valid
if embeddings and text_chunks:
    # Define the index name
    index_name = "medical"

    # Initialize Pinecone
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # Check if index exists
        existing_indexes = pc.list_indexes()
        if index_name not in [index.name for index in existing_indexes]:
            pc.create_index(
                name=index_name,
                dimension=384,  # Adjust dimension according to your embedding model
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  # Change region if needed
                ),
            )
            print(f"Index '{index_name}' created successfully")
        else:
            print(f"Index '{index_name}' already exists")

        # Store embeddings in Pinecone using LangChain's PineconeVectorStore
        docsearch = PineconeVectorStore.from_documents(
            documents=text_chunks,
            index_name=index_name,
            embedding=embeddings,
        )
        print("Embeddings successfully stored in Pinecone!")

    except Exception as e:
        print(f"Error interacting with Pinecone: {e}")
else:
    print("Skipping Pinecone storage due to missing data or embeddings.")


# from langchain_pinecone import PineconeVectorStore
# # Embed each chunk and upsert the embeddings into your Pinecone index.
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )

# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":5})

# retrieved_docs = retriever.invoke("medicines for fever.")
# for docs in retrieved_docs:
#     print(docs.page_content)
#     print('------------------------------------------------------------------------------------------------------------------------------------------------------')