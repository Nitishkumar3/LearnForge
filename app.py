"""
Core RAG Application - Flask Backend

A persistent, multi-document RAG system with intelligent retrieval.
Designed for single-user now, scalable to multi-user later.
"""

import os
import uuid
import json
from flask import Flask, request, jsonify, render_template, Response, current_app
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
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
from services.startup import run_startup_checks
from services import vector_store
from services import embedding
from services.retrieval import retrieve, invalidate_bm25_cache
from services import generation
from services import llm_providers as llm
from services.database import init_db
from services import database as db
from services import auth
from services.auth import auth_required
from services import email
from services import storage
from processors import pdf_processor
from utils import chunking

# ===================
# STARTUP HEALTH CHECKS
# ===================
import os as _os
if _os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
    if not run_startup_checks():
        import sys
        sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = UPLOADS_PATH

# Initialize PostgreSQL database (creates tables if not exist)
init_db()

# Chat history (in-memory for simplicity, could be persisted)
chat_history = []


# ===================
# ROUTES
# ===================

@app.route('/')
def index():
    """Render landing page."""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Render the main app dashboard."""
    return render_template('dashboard.html')




# ===================
# AUTH ROUTES
# ===================

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    """Register new user."""
    data = request.json

    # Validation
    required = ['name', 'email', 'password', 'confirm_password']
    for field in required:
        if not data.get(field):
            return jsonify({'error': f'{field} is required'}), 400

    if data['password'] != data['confirm_password']:
        return jsonify({'error': 'Passwords do not match'}), 400

    if len(data['password']) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400

    # Check if email exists
    if db.get_user_by_email(data['email']):
        return jsonify({'error': 'Email already registered'}), 400

    # Create user
    verification_token = auth.generate_verification_token()

    user = db.create_user(
        name=data['name'],
        email=data['email'].lower(),
        password_hash=auth.hash_password(data['password']),
        verification_token=verification_token
    )

    # Create default workspace
    db.create_workspace(
        user_id=str(user['id']),
        name='My First Workspace',
        description='Default workspace'
    )

    # Send verification email
    email.send_verification_email(
        to_email=user['email'],
        name=user['name'],
        token=verification_token
    )

    return jsonify({
        'success': True,
        'message': 'Account created. Please check your email to verify your account.',
        'user': {
            'id': str(user['id']),
            'name': user['name'],
            'email': user['email']
        }
    }), 201


@app.route('/api/auth/signin', methods=['POST'])
def signin():
    """Sign in user."""
    data = request.json

    if not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Email and password required'}), 400

    user = db.get_user_by_email(data['email'])

    if not user:
        return jsonify({'error': 'Invalid email or password'}), 401

    if not auth.verify_password(data['password'], user['password_hash']):
        return jsonify({'error': 'Invalid email or password'}), 401

    if not user['email_verified']:
        return jsonify({'error': 'Please verify your email first'}), 401

    # Generate token
    token = auth.generate_token(str(user['id']))

    return jsonify({
        'success': True,
        'token': token,
        'user': {
            'id': str(user['id']),
            'name': user['name'],
            'email': user['email']
        }
    })


@app.route('/api/auth/verify-email/<token>', methods=['GET'])
def verify_email(token):
    """Verify email with token."""
    user = db.get_user_by_verification_token(token)

    if not user:
        return jsonify({'error': 'Invalid verification link'}), 400

    db.verify_user_email(str(user['id']))

    return jsonify({
        'success': True,
        'message': 'Email verified successfully. You can now sign in.'
    })


@app.route('/api/auth/forgot-password', methods=['POST'])
def forgot_password():
    """Request password reset."""
    data = request.json
    user_email = data.get('email', '').lower()

    if not user_email:
        return jsonify({'error': 'Email is required'}), 400

    user = db.get_user_by_email(user_email)

    # Always return success to prevent email enumeration
    if not user:
        return jsonify({
            'success': True,
            'message': 'If an account exists, a reset link has been sent.'
        })

    # Generate reset token
    reset_token = auth.generate_reset_token()
    expires = datetime.utcnow() + timedelta(hours=1)

    db.set_reset_token(str(user['id']), reset_token, expires)

    # Send reset email
    email.send_password_reset_email(
        to_email=user['email'],
        name=user['name'],
        token=reset_token
    )

    return jsonify({
        'success': True,
        'message': 'If an account exists, a reset link has been sent.'
    })


@app.route('/api/auth/reset-password', methods=['POST'])
def reset_password():
    """Reset password with token."""
    data = request.json

    if not data.get('token') or not data.get('password'):
        return jsonify({'error': 'Token and password required'}), 400

    if data['password'] != data.get('confirm_password'):
        return jsonify({'error': 'Passwords do not match'}), 400

    if len(data['password']) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400

    user = db.get_user_by_reset_token(data['token'])

    if not user:
        return jsonify({'error': 'Invalid or expired reset link'}), 400

    # Check expiry
    if user['reset_token_expires']:
        expires = user['reset_token_expires']
        # Handle both datetime and string
        if isinstance(expires, str):
            expires = datetime.fromisoformat(expires.replace('Z', '+00:00'))
        if expires.replace(tzinfo=None) < datetime.utcnow():
            return jsonify({'error': 'Reset link has expired'}), 400

    db.update_password(str(user['id']), auth.hash_password(data['password']))

    return jsonify({
        'success': True,
        'message': 'Password reset successfully. You can now sign in.'
    })


@app.route('/api/auth/me', methods=['GET'])
@auth_required
def get_current_user():
    """Get current authenticated user."""
    user = db.get_user_by_id(request.user_id)

    if not user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({
        'user': {
            'id': str(user['id']),
            'name': user['name'],
            'email': user['email'],
            'email_verified': user['email_verified']
        }
    })


# ===================
# WORKSPACE ROUTES
# ===================

@app.route('/api/workspaces', methods=['GET'])
@auth_required
def list_workspaces():
    """List all workspaces for the authenticated user."""
    workspaces = db.get_workspaces_by_user(request.user_id)

    return jsonify({
        'workspaces': [{
            'id': str(w['id']),
            'name': w['name'],
            'description': w['description'],
            'created_at': w['created_at'].isoformat() if w['created_at'] else None
        } for w in workspaces]
    })


@app.route('/api/workspaces', methods=['POST'])
@auth_required
def create_workspace():
    """Create a new workspace."""
    data = request.json

    if not data.get('name'):
        return jsonify({'error': 'Workspace name is required'}), 400

    workspace = db.create_workspace(
        user_id=request.user_id,
        name=data['name'],
        description=data.get('description')
    )

    return jsonify({
        'success': True,
        'workspace': {
            'id': str(workspace['id']),
            'name': workspace['name'],
            'description': workspace['description'],
            'created_at': workspace['created_at'].isoformat() if workspace['created_at'] else None
        }
    }), 201


@app.route('/api/workspaces/<workspace_id>', methods=['GET'])
@auth_required
def get_workspace(workspace_id):
    """Get a specific workspace."""
    workspace = db.get_workspace_by_id_and_user(workspace_id, request.user_id)

    if not workspace:
        return jsonify({'error': 'Workspace not found'}), 404

    return jsonify({
        'workspace': {
            'id': str(workspace['id']),
            'name': workspace['name'],
            'description': workspace['description'],
            'created_at': workspace['created_at'].isoformat() if workspace['created_at'] else None
        }
    })


@app.route('/api/workspaces/<workspace_id>', methods=['PUT'])
@auth_required
def update_workspace_route(workspace_id):
    """Update a workspace."""
    data = request.json

    workspace = db.update_workspace(
        workspace_id=workspace_id,
        user_id=request.user_id,
        name=data.get('name'),
        description=data.get('description')
    )

    if not workspace:
        return jsonify({'error': 'Workspace not found'}), 404

    return jsonify({
        'success': True,
        'workspace': {
            'id': str(workspace['id']),
            'name': workspace['name'],
            'description': workspace['description']
        }
    })


@app.route('/api/workspaces/<workspace_id>', methods=['DELETE'])
@auth_required
def delete_workspace_route(workspace_id):
    """Delete a workspace and all its documents, S3 files, and vector embeddings."""
    # Check workspace exists and belongs to user
    workspace = db.get_workspace_by_id_and_user(workspace_id, request.user_id)

    if not workspace:
        return jsonify({'error': 'Workspace not found'}), 404

    # Check if this is the only workspace
    all_workspaces = db.get_workspaces_by_user(request.user_id)
    if len(all_workspaces) <= 1:
        return jsonify({'error': 'Cannot delete your only workspace'}), 400

    # Get all documents in this workspace before deletion
    documents = db.get_documents_by_workspace(workspace_id, request.user_id)

    # Delete vector embeddings for each document
    for doc in documents:
        try:
            # ChromaDB stores chunks with document_id in metadata
            vector_store.delete_document(doc['filename'])
        except Exception as e:
            print(f"Error deleting vectors for doc {doc['id']}: {e}")

    # Invalidate BM25 cache since documents are being removed
    invalidate_bm25_cache()

    # Delete S3 folder for this workspace (all files under user_id/workspace_id/)
    try:
        folder_prefix = f"{request.user_id}/{workspace_id}/"
        storage.delete_folder(folder_prefix)
    except Exception as e:
        print(f"Error deleting S3 folder: {e}")

    # Delete workspace from DB (cascades to documents table)
    db.delete_workspace(workspace_id, request.user_id)

    return jsonify({
        'success': True,
        'message': 'Workspace deleted'
    })


# ===================
# DOCUMENT ROUTES
# ===================

@app.route('/api/documents', methods=['GET'])
@auth_required
def list_documents():
    """List documents for the current user/workspace."""
    workspace_id = request.args.get('workspace_id')

    if workspace_id:
        # Verify workspace belongs to user
        workspace = db.get_workspace_by_id_and_user(workspace_id, request.user_id)
        if not workspace:
            return jsonify({"error": "Workspace not found"}), 404
        # Get documents for specific workspace
        documents = db.get_documents_by_workspace(workspace_id, request.user_id)
    else:
        # Get all documents for user
        documents = db.get_documents_by_user(request.user_id)

    # Format documents for response (convert datetime, uuid to strings)
    formatted_docs = []
    for doc in documents:
        formatted_docs.append({
            "id": str(doc['id']),
            "user_id": str(doc['user_id']),
            "workspace_id": str(doc['workspace_id']),
            "filename": doc['original_filename'],
            "storage_key": doc['storage_key'],
            "storage_bucket": doc['storage_bucket'],
            "num_pages": doc['num_pages'],
            "num_chunks": doc['num_chunks'],
            "file_size_bytes": doc['file_size_bytes'],
            "status": doc['status'],
            "upload_time": doc['created_at'].isoformat() if doc['created_at'] else None
        })

    return jsonify({
        "documents": formatted_docs,
        "total": len(formatted_docs),
        "total_chunks": vector_store.count()
    })


@app.route('/api/upload', methods=['POST'])
@auth_required
def upload_document():
    """
    Upload and process a PDF document.

    Expects:
        - file: PDF file (multipart/form-data)
        - workspace_id: Workspace to upload to

    Flow:
        1. Save file temporarily for PDF processing
        2. Extract text and generate embeddings
        3. Upload to S3 (if configured)
        4. Delete temp file
        5. Store metadata in registry
    """
    global chat_history

    # Validate request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    workspace_id = request.form.get('workspace_id')

    if not workspace_id:
        return jsonify({"error": "Workspace ID is required"}), 400

    # Verify workspace belongs to user
    workspace = db.get_workspace_by_id_and_user(workspace_id, request.user_id)
    if not workspace:
        return jsonify({"error": "Workspace not found"}), 404

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    temp_file_path = None

    try:
        # Generate document ID
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        original_filename = file.filename
        safe_filename = secure_filename(file.filename)

        # Save file temporarily for PDF processing (required for text extraction)
        temp_file_path = os.path.join(UPLOADS_PATH, f"temp_{doc_id}_{safe_filename}")
        file.save(temp_file_path)

        # Extract text from PDF
        pdf_result = pdf_processor.process(temp_file_path)

        # Chunk the text with metadata (include user_id and workspace_id)
        chunks = chunking.chunk_text(
            text=pdf_result["text"],
            document_id=doc_id,
            document_name=original_filename,
            document_type="pdf"
        )

        if not chunks:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return jsonify({"error": "Could not extract text from PDF"}), 400

        # Add user_id and workspace_id to all chunk metadata
        for chunk in chunks:
            chunk["metadata"]["user_id"] = request.user_id
            chunk["metadata"]["workspace_id"] = workspace_id

        # Generate embeddings using local server
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embedding.embed_texts(chunk_texts)

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

        # Upload to S3 (required)
        storage_key = storage.generate_key(
            user_id=request.user_id,
            workspace_id=workspace_id,
            filename=original_filename,
            doc_id=doc_id
        )
        storage_bucket = storage.BUCKET_NAME

        # Upload file to S3
        storage.upload_file(
            key=storage_key,
            file_path=temp_file_path,
            content_type='application/pdf'
        )

        # Delete temp file after S3 upload
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        temp_file_path = None

        # Register document in PostgreSQL
        doc = db.create_document(
            workspace_id=workspace_id,
            user_id=request.user_id,
            filename=doc_id,  # Internal filename with doc_id
            original_filename=original_filename,
            file_type='pdf',
            file_size_bytes=pdf_result["file_size"],
            storage_key=storage_key,
            storage_bucket=storage_bucket,
            num_pages=pdf_result["total_pages"],
            num_chunks=len(chunks),
            status='processed'
        )

        return jsonify({
            "success": True,
            "message": f"Document '{original_filename}' processed successfully!",
            "document": {
                "id": str(doc['id']),
                "filename": original_filename,
                "pages": pdf_result["total_pages"],
                "chunks": len(chunks),
                "storage": "s3"
            }
        })

    except Exception as e:
        # Clean up temp file on error
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return jsonify({"error": str(e)}), 500


@app.route('/api/documents/<doc_id>', methods=['DELETE'])
@auth_required
def delete_document(doc_id):
    """Delete a document from vector store, S3/local storage, and database."""
    try:
        # Get document info (with ownership check)
        doc = db.get_document_by_id_and_user(doc_id, request.user_id)

        if not doc:
            return jsonify({"error": "Document not found"}), 404

        # Delete from vector store (using the internal filename which contains doc_id pattern)
        # ChromaDB chunks are stored with document_id in metadata
        vector_store.delete_document(doc['filename'])

        # Invalidate BM25 cache for hybrid search
        invalidate_bm25_cache()

        # Delete from S3
        if doc.get('storage_key'):
            storage.delete_file(doc['storage_key'])

        # Delete from database (returns deleted doc for confirmation)
        db.delete_document(doc_id, request.user_id)

        return jsonify({
            "success": True,
            "message": f"Document '{doc.get('original_filename', doc_id)}' deleted"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
@auth_required
def chat():
    """
    Chat with the documents.

    Expects JSON:
        - question: User question
        - workspace_id: Workspace to search in
        - use_search: Enable web search (optional)
        - use_thinking: Enable thinking mode (optional)
    """
    global chat_history

    data = request.json
    question = data.get('question', '').strip()
    workspace_id = data.get('workspace_id')
    use_search = data.get('use_search', False)
    use_thinking = data.get('use_thinking', False)

    if not question:
        return jsonify({"error": "Question is required"}), 400

    if not workspace_id:
        return jsonify({"error": "Workspace ID is required"}), 400

    # Verify workspace belongs to user
    workspace = db.get_workspace_by_id_and_user(workspace_id, request.user_id)
    if not workspace:
        return jsonify({"error": "Workspace not found"}), 404

    # Check if we have any documents in this workspace
    user_docs = db.get_documents_by_workspace(workspace_id, request.user_id)
    if not user_docs:
        return jsonify({
            "error": "No documents uploaded. Please upload some PDFs first."
        }), 400

    try:
        # Retrieve relevant chunks filtered by user and workspace
        retrieval_result = retrieve(
            question,
            user_id=request.user_id, workspace_id=workspace_id
        )

        # Generate answer with optional features
        answer_result = generation.generate_answer(
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
@auth_required
def chat_stream():
    """
    Stream chat response using Server-Sent Events.

    Expects JSON:
        - question: User question
        - workspace_id: Workspace to search in
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
    workspace_id = data.get('workspace_id')
    use_search = data.get('use_search', False)
    use_thinking = data.get('use_thinking', False)

    if not question:
        return jsonify({"error": "Question is required"}), 400

    if not workspace_id:
        return jsonify({"error": "Workspace ID is required"}), 400

    # Verify workspace belongs to user
    workspace = db.get_workspace_by_id_and_user(workspace_id, request.user_id)
    if not workspace:
        return jsonify({"error": "Workspace not found"}), 404

    # Check if we have any documents in this workspace
    user_docs = db.get_documents_by_workspace(workspace_id, request.user_id)
    if not user_docs:
        return jsonify({
            "error": "No documents uploaded. Please upload some PDFs first."
        }), 400

    # Capture user_id for the closure
    user_id = request.user_id

    def generate():
        try:
            # Retrieve relevant chunks filtered by user and workspace
            retrieval_result = retrieve(
                question,
                user_id=user_id, workspace_id=workspace_id
            )

            # Stream answer
            full_answer = ""
            sources = []

            for event in generation.generate_answer_stream(
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
        current_model = llm.get_current_model()
        if current_model.get("type") != "image":
            return jsonify({"error": f"Current model '{current_model['name']}' does not support image generation"}), 400

        result = llm.generate_image(prompt, aspect_ratio=aspect_ratio)

        return jsonify({
            "success": True,
            "image_base64": result["image_base64"],
            "mime_type": result["mime_type"],
            "prompt": prompt,
            "aspect_ratio": aspect_ratio
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
@auth_required
def get_stats():
    """Get statistics for user's workspace."""
    workspace_id = request.args.get('workspace_id')

    # Filter documents by user and workspace
    if workspace_id:
        documents = db.get_documents_by_workspace(workspace_id, request.user_id)
    else:
        documents = db.get_documents_by_user(request.user_id)

    total_pages = sum(d.get("num_pages", 0) or 0 for d in documents)
    total_size = sum(d.get("file_size_bytes", 0) or 0 for d in documents)

    return jsonify({
        "total_documents": len(documents),
        "total_chunks": sum(d.get("num_chunks", 0) or 0 for d in documents),
        "total_pages": total_pages,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "chat_history_length": len(chat_history)
    })


@app.route('/api/models', methods=['GET'])
def list_models():
    """List all available LLM models."""
    return jsonify({
        "models": llm.list_models(),
        "current": llm.get_current_model()
    })


@app.route('/api/models/switch', methods=['POST'])
def switch_model():
    """Switch to a different LLM model."""
    data = request.json
    model_name = data.get('model')

    if not model_name:
        return jsonify({"error": "Model name required"}), 400

    if llm.set_model(model_name):
        return jsonify({
            "success": True,
            "message": f"Switched to {model_name}",
            "current": llm.get_current_model()
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


# ===================
# AUTH PAGE ROUTES
# ===================

@app.route('/signin')
def signin_page():
    """Render sign in page."""
    return render_template('signin.html')


@app.route('/signup')
def signup_page():
    """Render sign up page."""
    return render_template('signup.html')


@app.route('/forgotpassword')
def forgotpassword_page():
    """Render forgot password page."""
    return render_template('forgotpassword.html')


@app.route('/resetpassword')
def resetpassword_page():
    """Render reset password page."""
    return render_template('resetpassword.html')


# ===================
# RUN APP
# ===================

if __name__ == '__main__':
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG
    )
