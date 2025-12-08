"""
Core RAG Application - Flask Backend

A persistent, multi-document RAG system with intelligent retrieval.
Designed for single-user now, scalable to multi-user later.
"""

import os
import uuid
import json
from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===================
# CONFIGURATION
# ===================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
UPLOADS_PATH = os.path.join(DATA_DIR, "uploads")

MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", "104857600"))  # 100MB
ALLOWED_EXTENSIONS = {'pdf'}

FLASK_HOST = os.getenv("FLASK_APP_HOST", "127.0.0.1")
FLASK_PORT = int(os.getenv("FLASK_APP_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"

EMBEDDING_SERVER_URL = os.getenv("EMBEDDING_SERVER_URL", "http://localhost:5001")

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs(UPLOADS_PATH, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ===================
# IMPORTS (after config so paths are ready)
# ===================
from services.vector_store import VectorStore, DocumentRegistry
from services.embedding import EmbeddingService
from services.retrieval import retrieve, invalidate_bm25_cache
from services.generation import GenerationService
from services.llm_providers import get_llm_manager
from processors.pdf_processor import PDFProcessor
from utils.chunking import TextChunker

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = UPLOADS_PATH

# Initialize persistent services
vector_store = VectorStore()
document_registry = DocumentRegistry()
pdf_processor = PDFProcessor()
text_chunker = TextChunker()

# Initialize services (all config loaded from .env)
embedding_service = EmbeddingService()
generation_service = GenerationService()
llm_manager = get_llm_manager()

# Chat history (in-memory for simplicity, could be persisted)
chat_history = []


# ===================
# ROUTES
# ===================

@app.route('/')
def index():
    """Render the main UI."""
    return render_template('index.html')


@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List all uploaded documents."""
    documents = document_registry.list_documents()

    return jsonify({
        "documents": documents,
        "total": len(documents),
        "total_chunks": vector_store.count()
    })


@app.route('/api/upload', methods=['POST'])
def upload_document():
    """
    Upload and process a PDF document.

    Expects:
        - file: PDF file (multipart/form-data)

    Note: Embeddings are generated using local server (no API key needed)
    """
    global chat_history

    # Validate request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    try:
        # Generate document ID
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"

        # Secure and save file
        original_filename = file.filename
        safe_filename = secure_filename(file.filename)
        stored_filename = f"{doc_id}_{safe_filename}"
        file_path = os.path.join(UPLOADS_PATH, stored_filename)
        file.save(file_path)

        # Extract text from PDF
        pdf_result = pdf_processor.process(file_path)

        # Chunk the text with metadata
        chunks = text_chunker.chunk_text(
            text=pdf_result["text"],
            document_id=doc_id,
            document_name=original_filename,
            document_type="pdf"
        )

        if not chunks:
            os.remove(file_path)
            return jsonify({"error": "Could not extract text from PDF"}), 400

        # Generate embeddings using local server
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embedding_service.embed_texts(chunk_texts)

        # Prepare metadata for storage
        metadatas = [c["metadata"] for c in chunks]

        # Store in vector database
        vector_store.add_chunks(
            chunks=chunk_texts,
            embeddings=embeddings,
            metadatas=metadatas,
            document_id=doc_id
        )

        # Invalidate BM25 cache for hybrid search
        invalidate_bm25_cache()

        # Register document
        doc_info = {
            "id": doc_id,
            "filename": original_filename,
            "stored_filename": stored_filename,
            "file_path": file_path,
            "num_pages": pdf_result["total_pages"],
            "num_chunks": len(chunks),
            "file_size_bytes": pdf_result["file_size"],
            "status": "processed",
            "upload_time": datetime.now().isoformat()
        }
        document_registry.add_document(doc_info)

        return jsonify({
            "success": True,
            "message": f"Document '{original_filename}' processed successfully!",
            "document": {
                "id": doc_id,
                "filename": original_filename,
                "pages": pdf_result["total_pages"],
                "chunks": len(chunks)
            }
        })

    except Exception as e:
        # Clean up on error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": str(e)}), 500


@app.route('/api/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a document and its chunks."""
    try:
        # Get document info
        doc = document_registry.get_document(doc_id)

        if not doc:
            return jsonify({"error": "Document not found"}), 404

        # Delete from vector store
        vector_store.delete_document(doc_id)

        # Invalidate BM25 cache for hybrid search
        invalidate_bm25_cache()

        # Delete file
        if "file_path" in doc and os.path.exists(doc["file_path"]):
            os.remove(doc["file_path"])

        # Delete from registry
        document_registry.delete_document(doc_id)

        return jsonify({
            "success": True,
            "message": f"Document '{doc.get('filename', doc_id)}' deleted"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat with the documents.

    Expects JSON:
        - question: User question
        - use_search: Enable web search (optional)
        - use_thinking: Enable thinking mode (optional)
    """
    global chat_history

    data = request.json
    question = data.get('question', '').strip()
    use_search = data.get('use_search', False)
    use_thinking = data.get('use_thinking', False)

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Check if we have any documents
    if vector_store.count() == 0:
        return jsonify({
            "error": "No documents uploaded. Please upload some PDFs first."
        }), 400

    try:
        # Retrieve relevant chunks
        retrieval_result = retrieve(vector_store, embedding_service, question, llm_manager)

        # Generate answer with optional features
        answer_result = generation_service.generate_answer(
            query=question,
            chunks=retrieval_result["chunks"],
            metadatas=retrieval_result["metadatas"],
            chat_history=chat_history,
            use_search=use_search,
            use_thinking=use_thinking
        )

        # Update chat history
        chat_history.append({
            "question": question,
            "answer": answer_result["answer"]
        })

        # Keep only last 20 exchanges
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]

        return jsonify({
            "answer": answer_result["answer"],
            "sources": answer_result["sources"],
            "chunks_used": answer_result["chunks_used"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """
    Stream chat response using Server-Sent Events.

    Expects JSON:
        - question: User question
        - use_search: Enable web search (optional)
        - use_thinking: Enable thinking mode (optional)

    Returns SSE stream with events:
        - data: {"type": "chunk", "content": "..."} for text chunks
        - data: {"type": "done", "content": "...", "sources": [...]} when complete
        - data: {"type": "error", "message": "..."} on error
    """
    global chat_history

    data = request.json
    question = data.get('question', '').strip()
    use_search = data.get('use_search', False)
    use_thinking = data.get('use_thinking', False)

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Check if we have any documents
    if vector_store.count() == 0:
        return jsonify({
            "error": "No documents uploaded. Please upload some PDFs first."
        }), 400

    def generate():
        try:
            # Retrieve relevant chunks
            retrieval_result = retrieve(vector_store, embedding_service, question, llm_manager)

            # Stream answer
            full_answer = ""
            sources = []

            for event in generation_service.generate_answer_stream(
                query=question,
                chunks=retrieval_result["chunks"],
                metadatas=retrieval_result["metadatas"],
                chat_history=chat_history,
                use_search=use_search,
                use_thinking=use_thinking
            ):
                if event["type"] == "thinking":
                    yield f"data: {json.dumps(event)}\n\n"
                elif event["type"] == "chunk":
                    yield f"data: {json.dumps(event)}\n\n"
                elif event["type"] == "done":
                    full_answer = event.get("content", "")
                    sources = event.get("sources", [])
                    yield f"data: {json.dumps(event)}\n\n"
                elif event["type"] == "error":
                    yield f"data: {json.dumps(event)}\n\n"
                    return

            # Update chat history after streaming complete
            if full_answer:
                chat_history.append({
                    "question": question,
                    "answer": full_answer
                })

                # Keep only last 20 exchanges
                if len(chat_history) > 20:
                    chat_history[:] = chat_history[-20:]

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/generate-image', methods=['POST'])
def generate_image():
    """
    Generate an image using Imagen model.

    Expects JSON:
        - prompt: Image description/prompt
        - aspect_ratio: Optional aspect ratio (1:1, 9:16, 16:9, 4:3, 3:4)

    Returns:
        - image_base64: Base64 encoded image
        - mime_type: Image MIME type
    """
    data = request.json
    prompt = data.get('prompt', '').strip()
    aspect_ratio = data.get('aspect_ratio', '1:1')

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # Validate aspect ratio
    valid_ratios = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5"]
    if aspect_ratio not in valid_ratios:
        return jsonify({"error": f"Invalid aspect ratio. Must be one of: {valid_ratios}"}), 400

    try:
        # Check if current model supports image generation
        current_model = llm_manager.get_current_model()
        if current_model.get("type") != "image":
            return jsonify({"error": f"Current model '{current_model['name']}' does not support image generation"}), 400

        result = llm_manager.generate_image(prompt, aspect_ratio=aspect_ratio)

        return jsonify({
            "success": True,
            "image_base64": result["image_base64"],
            "mime_type": result["mime_type"],
            "prompt": prompt,
            "aspect_ratio": aspect_ratio
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/clear', methods=['POST'])
def clear_all():
    """Clear all documents and chat history."""
    global chat_history

    try:
        # Clear vector store
        vector_store.clear_all()

        # Invalidate BM25 cache for hybrid search
        invalidate_bm25_cache()

        # Clear document registry
        document_registry.clear_all()

        # Clear uploaded files
        for filename in os.listdir(UPLOADS_PATH):
            file_path = os.path.join(UPLOADS_PATH, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Clear chat history
        chat_history = []

        return jsonify({
            "success": True,
            "message": "All data cleared successfully"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """Clear only chat history."""
    global chat_history
    chat_history = []
    return jsonify({
        "success": True,
        "message": "Chat history cleared"
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    documents = document_registry.list_documents()

    total_pages = sum(d.get("num_pages", 0) for d in documents)
    total_size = sum(d.get("file_size_bytes", 0) for d in documents)

    return jsonify({
        "total_documents": len(documents),
        "total_chunks": vector_store.count(),
        "total_pages": total_pages,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "chat_history_length": len(chat_history)
    })


@app.route('/api/models', methods=['GET'])
def list_models():
    """List all available LLM models."""
    return jsonify({
        "models": llm_manager.list_models(),
        "current": llm_manager.get_current_model()
    })


@app.route('/api/models/switch', methods=['POST'])
def switch_model():
    """Switch to a different LLM model."""
    data = request.json
    model_name = data.get('model')

    if not model_name:
        return jsonify({"error": "Model name required"}), 400

    if llm_manager.set_model(model_name):
        return jsonify({
            "success": True,
            "message": f"Switched to {model_name}",
            "current": llm_manager.get_current_model()
        })
    else:
        return jsonify({"error": f"Unknown model: {model_name}"}), 400


# ===================
# ERROR HANDLERS
# ===================

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    max_mb = MAX_CONTENT_LENGTH / (1024 * 1024)
    return jsonify({
        "error": f"File too large. Maximum size is {max_mb}MB"
    }), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    return jsonify({
        "error": "An internal error occurred. Please try again."
    }), 500


# ===================
# RUN
# ===================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Core RAG System")
    print("="*50)
    print(f"Data directory: {DATA_DIR}")
    print(f"ChromaDB path: {CHROMA_DB_PATH}")
    print(f"Embedding server: {EMBEDDING_SERVER_URL}")
    print(f"Documents: {len(document_registry.list_documents())}")
    print(f"Total chunks: {vector_store.count()}")
    print("="*50 + "\n")

    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG
    )
