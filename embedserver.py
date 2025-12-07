#$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
from flask import Flask, request, jsonify
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and tokenizer
session = None
tokenizer = None

def load_model():
    """Load ONNX model and tokenizer at startup"""
    global session, tokenizer

    try:
        logger.info("Loading model...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the project root and then to bge-m3-onnx folder
        model_path = os.path.join(script_dir, "bge-m3-onnx", "model.onnx")
        model_path = os.path.normpath(model_path)

        logger.info(f"Model path: {model_path}")

        session = ort.InferenceSession(
            model_path,
            providers=providers
        )
        
        logger.info(f"Active providers: {session.get_providers()}")
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        
        logger.info("Model and tokenizer loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def generate_embedding(text):
    """Generate embedding for a single text"""
    try:
        # Tokenize
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="np")
        
        # Convert to int64
        for key in inputs:
            if inputs[key].dtype == np.int32:
                inputs[key] = inputs[key].astype(np.int64)
        
        # Run inference
        outputs = session.run(None, dict(inputs))
        dense_embedding = outputs[0]
        
        return dense_embedding[0].tolist()  # Convert to list for JSON
        
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": "BGE-M3",
        "providers": session.get_providers() if session else []
    })

@app.route('/embed', methods=['POST'])
def embed():
    """
    Generate embeddings for text(s)
    
    Request body:
    {
        "text": "single text"  # or
        "texts": ["text1", "text2", ...]  # for batch
    }
    
    Response:
    {
        "embedding": [...],  # for single text
        "embeddings": [[...], [...]],  # for batch
        "dimension": 1024
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Single text
        if "text" in data:
            text = data["text"]
            if not text or not isinstance(text, str):
                return jsonify({"error": "Invalid text format"}), 400
            
            embedding = generate_embedding(text)
            
            return jsonify({
                "embedding": embedding,
                "dimension": len(embedding)
            })
        
        # Batch texts
        elif "texts" in data:
            texts = data["texts"]
            if not texts or not isinstance(texts, list):
                return jsonify({"error": "Invalid texts format"}), 400
            
            if not all(isinstance(t, str) for t in texts):
                return jsonify({"error": "All texts must be strings"}), 400
            
            embeddings = [generate_embedding(text) for text in texts]
            
            return jsonify({
                "embeddings": embeddings,
                "dimension": len(embeddings[0]) if embeddings else 0,
                "count": len(embeddings)
            })
        
        else:
            return jsonify({"error": "Either 'text' or 'texts' field required"}), 400
    
    except Exception as e:
        logger.error(f"Error in /embed endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    """API documentation"""
    return jsonify({
        "name": "BGE-M3 Embedding API",
        "version": "1.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/embed": "POST - Generate embeddings",
            "/": "GET - API documentation"
        },
        "usage": {
            "single": {
                "url": "/embed",
                "method": "POST",
                "body": {"text": "your text here"}
            },
            "batch": {
                "url": "/embed",
                "method": "POST",
                "body": {"texts": ["text1", "text2"]}
            }
        }
    })

if __name__ == '__main__':
    # Load model at startup
    load_model()

    # Run Flask app
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5001,  # Changed from 5000 to avoid conflict with main app
        debug=False  # Set to False for production
    )