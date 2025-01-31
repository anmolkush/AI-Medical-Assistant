from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone  # Updated import
from pinecone import Pinecone as PineconeClient  # Corrected Pinecone import
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# Load the embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone with the new method
pinecone_client = PineconeClient(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# Define the index name
index_name = "medical"

# Load the existing index (skip index creation)
docsearch = Pinecone.from_existing_index(index_name, embeddings)

# Define the prompt template
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Initialize the LLM model using Groq's API
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.6,
    max_retries=2,
)

# Create the retrieval QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Route to render the chat page
@app.route("/")
def index():
    return render_template('chat.html')

# Route to handle user queries
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"User input: {msg}")
    result = qa.invoke({"query": msg})
    print(f"Response: {result['result']}")
    return str(result["result"])

# Run the Flask application
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
