from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import json
import numpy as np
from groq import Groq
import random
import os
from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer
from embed import generate_query_embedding
# Load environment variables
load_dotenv()

# Retrieve Groq API keys
GROQ_API_KEYS = [
    "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",
]

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for React frontend

# Load FAISS index and metadata
index = faiss.read_index("email_faiss_index")
with open("email_metadata.json", "r", encoding="utf-8") as meta_file:
    metadata = json.load(meta_file)

# Preprocess emails and load the documents
def preprocess_emails(email_data):
    documents = []
    for email in email_data:
        # Combine subject and body for the embedding
        combined_content = f"Subject: {email['subject']}\n\n{email['body']}"
        documents.append(combined_content)
    return documents

# Load the email JSON data
with open("cleaned_email_dataset.json", "r", encoding="utf-8") as file:
    email_data = json.load(file)

# Preprocess the email data into documents
documents = preprocess_emails(email_data)

# Function to query FAISS index
def query_index(query_embedding, top_k=19):
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(metadata):
            results.append({
                "metadata": metadata[idx],
                "distance": float(dist),
                "content": documents[idx][:500]
            })
    return results

# Function to generate response with Groq API
def generate_response_with_groq(context, query):
    try:
        # Select a Groq API key
        groq_api = random.choice(GROQ_API_KEYS)
        client = Groq(api_key=groq_api)

        # Build the prompt for Groq
        generation_prompt = (
            f"You are an expert email assistant. Use the following context to answer the query. "
            f"Provide a concise and accurate response.\n\nContext:\n{context}\n\nQuery: {query}\n\nAnswer:"
        )

        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": generation_prompt}],
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=2048,
            stream=False,
        )

        # Retrieve the response
        response = chat_completion.choices[0].message.content
        return response

    except Exception as e:
        return f"Error generating response with Groq: {str(e)}"

# Load embedding model
# model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # Pre-trained model for embeddings

# API endpoint for processing user queries
@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Parse JSON request
        data = request.get_json()
        query = data.get("query", "").strip() # type: ignore

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # **Embedding-Based Search Query**
        # Generate query embedding
        # Generate query embedding
        query_embedding = generate_query_embedding(query)
        # query_embedding = model.encode([query], convert_to_numpy=True)

        # Retrieve top results from FAISS
        results = query_index(query_embedding)

        # Combine retrieved results into context
        if results:
            context = "\n\n".join([result["content"] for result in results])
        else:
            context = ""

        # Generate response using Groq
        response = generate_response_with_groq(context, query) if context else "No relevant emails found."

        # Return the response as JSON
        return jsonify({
            "query": query,
            "response": response,
            "retrieved_emails": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API endpoint to showcase emails
@app.route('/api/emails', methods=['GET'])
def get_emails():
    try:
        # Load the email dataset from the JSON file
        with open("email_dataset.json", "r", encoding="utf-8") as file:
            emails = json.load(file)

        return jsonify(emails)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
