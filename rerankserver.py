from flask import Flask, request, jsonify
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

session = None
tokenizer = None

def load_model():
    global session, tokenizer

    try:
        logger.info("Loading reranker model...")

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "ms-marco-MiniLM-L6-v2-onnx", "model.onnx")
        model_path = os.path.normpath(model_path)

        logger.info(f"Model path: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Download from: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/main/onnx"
            )

        session = ort.InferenceSession(
            model_path,
            providers=providers
        )

        logger.info(f"Active providers: {session.get_providers()}")

        logger.info("Loading tokenizer...")
        tokenizer_path = os.path.join(script_dir, "ms-marco-MiniLM-L6-v2-onnx")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        logger.info("Reranker model and tokenizer loaded successfully!")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def compute_relevance_score(query, document):
    """
    Compute relevance score for a query-document pair.

    Cross-encoder processes query and document together,
    outputting a relevance score.
    """
    try:
        # Tokenize query-document pair together
        inputs = tokenizer(
            query,
            document,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )

        # Convert to int64 if needed
        for key in inputs:
            if inputs[key].dtype == np.int32:
                inputs[key] = inputs[key].astype(np.int64)

        # Run inference
        outputs = session.run(None, dict(inputs))

        # Output is logits - single score for relevance
        # For cross-encoder, output shape is typically (batch_size, 1) or (batch_size,)
        logits = outputs[0]

        # Extract scalar score
        if len(logits.shape) > 1:
            score = float(logits[0][0])
        else:
            score = float(logits[0])

        return score

    except Exception as e:
        logger.error(f"Error computing score: {str(e)}")
        raise

def batch_compute_scores(query, documents):
    """
    Compute relevance scores for multiple documents against a query.

    Batches all query-document pairs for efficient GPU inference.
    """
    try:
        if not documents:
            return []

        # Tokenize all pairs at once
        pairs = [(query, doc) for doc in documents]

        inputs = tokenizer(
            [p[0] for p in pairs],  # queries
            [p[1] for p in pairs],  # documents
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )

        # Convert to int64 if needed
        for key in inputs:
            if inputs[key].dtype == np.int32:
                inputs[key] = inputs[key].astype(np.int64)

        # Run inference
        outputs = session.run(None, dict(inputs))
        logits = outputs[0]

        # Extract scores
        if len(logits.shape) > 1:
            scores = [float(logits[i][0]) for i in range(len(documents))]
        else:
            scores = [float(logits[i]) for i in range(len(documents))]

        return scores

    except Exception as e:
        logger.error(f"Error in batch scoring: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model": "ms-marco-MiniLM-L6-v2",
        "type": "cross-encoder-reranker",
        "providers": session.get_providers() if session else []
    })

@app.route('/rerank', methods=['POST'])
def rerank():
    """
    Rerank documents by relevance to query.

    Request body:
    {
        "query": "user question",
        "documents": ["doc1", "doc2", ...]
    }

    Response:
    {
        "results": [
            {"index": 0, "score": 0.95, "text": "doc1"},
            {"index": 1, "score": 0.72, "text": "doc2"},
            ...
        ]
    }

    Results are sorted by score (highest first).
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        query = data.get('query', '')
        documents = data.get('documents', [])

        if not query:
            return jsonify({"error": "Query is required"}), 400

        if not documents or not isinstance(documents, list):
            return jsonify({"error": "Documents list is required"}), 400

        # Batch compute all scores
        scores = batch_compute_scores(query, documents)

        # Build results with original indices
        results = [
            {"index": i, "score": scores[i], "text": documents[i]}
            for i in range(len(documents))
        ]

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return jsonify({
            "results": results,
            "query": query,
            "total": len(results)
        })

    except Exception as e:
        logger.error(f"Error in /rerank endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5002, debug=False)