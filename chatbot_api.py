import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings

# Load your HuggingFace token
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Init Flask app
app = Flask(__name__)
CORS(app)  # Allows access from React frontend

# Load the vector DB
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Load the LLM from HuggingFace
def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

# Set the custom prompt template
def set_custom_prompt():
    prompt_template = """
    Use the pieces of information provided in the context to answer user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Don't provide anything out of the given context.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Define the API endpoint
@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json.get("message", "")
    if not user_query:
        return jsonify({"error": "Empty query"}), 400

    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(),
            chain_type="stuff",
            retriever=get_vectorstore().as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt()}
        )

        response = qa_chain.invoke({'query': user_query})

        return jsonify({
            "response": response["result"],
            "sources": [doc.metadata for doc in response["source_documents"]]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask server
if __name__ == '__main__':
    app.run(port=5001)
