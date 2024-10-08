from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore

app = Flask(__name__)
CORS(app)

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

index = faiss.read_index("index.faiss")
with open("index.pkl", "rb") as f:
    docstore = pickle.load(f)

vector_store = FAISS(
    np.empty((0, model.config.hidden_size)),  
    index,
    docstore,
    {i: str(i) for i in range(len(docstore))}
)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    
    _, indices = vector_store.index.search(query_embedding, k=5)  
    
    results = [vector_store.docstore[i] for i in indices[0]]
    
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
