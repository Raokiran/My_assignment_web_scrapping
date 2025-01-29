import os
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# data scrapping
url = "https://brainlox.com/courses/category/technical"
loader = WebBaseLoader(url)
documents = loader.load()

# Splitting text for better embedding search
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splitted_docs = text_splitter.split_documents(documents)

# Step 2: Create and save FAISS vector database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_db = FAISS.from_documents(splitted_docs, embeddings)
faiss_db.save_local("vectorstore")  # Save FAISS index

# Function to handle queries
def get_response(query):
    if not os.path.exists("vectorstore"):
        return "Vector database not found. Run the script to generate it first."

    # Load stored FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

    # Perform similarity search
    docs = vectorstore.similarity_search(query, k=3)

    if not docs:
        return "No relevant information found."

    response = "\n\n".join([doc.page_content for doc in docs])
    return response


# Flask Application Setup with Flask-RESTful
app = Flask(__name__)
api = Api(app)

class Chat(Resource):
    def get(self):
        # Return a simple message on GET request
        return jsonify({"message": "Hello."})

    def post(self):
        # Handle POST requests to interact with the chatbot use the postman for post
        data = request.get_json()

        # Validate the query
        if not data or "query" not in data:
            return jsonify({"error": "Query is required"}), 400

        query = data["query"]
        response = get_response(query)
        return jsonify({"response": response})

# Add the Chat resource to the API
api.add_resource(Chat, '/chat')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
