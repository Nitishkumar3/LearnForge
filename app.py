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

FLASK_HOST = os.getenv("FLASK_APP_HOST", "127.0.0.1")
FLASK_PORT = int(os.getenv("FLASK_APP_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"

EMBEDDING_SERVER_URL = os.getenv("EMBEDDING_SERVER_URL", "http://localhost:5001")

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs(UPLOADS_PATH, exist_ok=True)

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
from processors import file_types
from utils import chunking


def allowed_file(filename):
    """Check if file extension is allowed."""
    return file_types.is_allowed_file(filename)

# ===================
# STARTUP HEALTH CHECKS
# ===================
import os as _os
if _os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
    if not run_startup_checks():
        import sys
        sys.exit(1)

# Initialize Flask app with static folder
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = UPLOADS_PATH

# Initialize PostgreSQL database (creates tables if not exist)
init_db()

# Current conversation tracking (per request - real history is in DB)


# ===================
# ROUTES
# ===================

@app.route('/')
def index():
    """Render landing page."""
    return render_template('index.html')


@app.route('/chat')
@app.route('/chat/<workspace_id>')
@app.route('/chat/<workspace_id>/<conversation_id>')
def chat(workspace_id=None, conversation_id=None):
    """Render the chat page. Optionally load a specific workspace/conversation."""
    return render_template('chat.html')


@app.route('/study')
@app.route('/study/<workspace_id>')
@app.route('/study/<workspace_id>/<int:module_id>')
@app.route('/study/<workspace_id>/<int:module_id>/<int:subtopic_index>')
def study(workspace_id=None, module_id=None, subtopic_index=None):
    """Render the study materials page. Optionally load specific content."""
    return render_template('study.html')


@app.route('/uploads')
@app.route('/uploads/<workspace_id>')
def uploads(workspace_id=None):
    """Render the uploads page."""
    return render_template('uploads.html')


@app.route('/view/<doc_id>')
def view_document(doc_id):
    """Render the document viewer page."""
    return render_template('view.html')


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
    """Delete a workspace and all its documents, conversations, S3 files, and vector embeddings."""
    # Check workspace exists and belongs to user
    workspace = db.get_workspace_by_id_and_user(workspace_id, request.user_id)

    if not workspace:
        return jsonify({'error': 'Workspace not found'}), 404

    # Check if this is the only workspace
    all_workspaces = db.get_workspaces_by_user(request.user_id)
    if len(all_workspaces) <= 1:
        return jsonify({'error': 'Cannot delete your only workspace'}), 400

    # Delete all documents (S3 + ChromaDB + DB)
    documents = db.get_documents_by_workspace(workspace_id, request.user_id)
    for doc in documents:
        try:
            vector_store.delete_document(doc['filename'])
        except Exception as e:
            print(f"Error deleting vectors for doc {doc['id']}: {e}")

    invalidate_bm25_cache()

    # Delete all conversations and their S3 images
    conversations = db.get_workspace_conversations(workspace_id, request.user_id)
    for conv in conversations:
        s3_keys = db.get_conversation_image_s3_keys(conv['id'])
        for key in s3_keys:
            try:
                storage.delete_file(key)
            except Exception as e:
                print(f"Failed to delete S3 image {key}: {e}")

    # Delete entire S3 folder for this workspace (documents + any remaining files)
    try:
        folder_prefix = f"{request.user_id}/{workspace_id}/"
        storage.delete_folder(folder_prefix)
    except Exception as e:
        print(f"Error deleting S3 folder: {e}")

    # Delete workspace from DB (cascades to documents and conversations)
    db.delete_workspace(workspace_id, request.user_id)

    return jsonify({
        'success': True,
        'message': 'Workspace deleted',
        'deleted_documents': len(documents),
        'deleted_conversations': len(conversations)
    })


# ===================
# SYLLABUS ROUTES
# ===================

@app.route('/api/workspaces/<workspace_id>/syllabus', methods=['PUT'])
@auth_required
def save_syllabus(workspace_id):
    """Parse and save syllabus for a workspace."""
    # Verify workspace belongs to user
    workspace = db.get_workspace_by_id_and_user(workspace_id, request.user_id)
    if not workspace:
        return jsonify({'error': 'Workspace not found'}), 404

    data = request.json
    syllabus_text = data.get('syllabus_text', '').strip()

    if not syllabus_text:
        return jsonify({'error': 'Syllabus text is required'}), 400

    # Parse syllabus using LLM
    parsed = generation.parse_syllabus(syllabus_text)

    if 'error' in parsed:
        return jsonify({'error': parsed['error']}), 400

    # Save to database
    db.update_workspace_syllabus(workspace_id, parsed)

    return jsonify({
        'success': True,
        'syllabus': parsed
    })


@app.route('/api/workspaces/<workspace_id>/syllabus', methods=['GET'])
@auth_required
def get_syllabus(workspace_id):
    """Get syllabus for a workspace."""
    # Verify workspace belongs to user
    workspace = db.get_workspace_by_id_and_user(workspace_id, request.user_id)
    if not workspace:
        return jsonify({'error': 'Workspace not found'}), 404

    syllabus = db.get_workspace_syllabus(workspace_id)

    return jsonify({
        'syllabus': syllabus
    })


@app.route('/api/workspaces/<workspace_id>/syllabus', methods=['DELETE'])
@auth_required
def delete_syllabus(workspace_id):
    """Clear syllabus from a workspace."""
    # Verify workspace belongs to user
    workspace = db.get_workspace_by_id_and_user(workspace_id, request.user_id)
    if not workspace:
        return jsonify({'error': 'Workspace not found'}), 404

    db.clear_workspace_syllabus(workspace_id)

    return jsonify({
        'success': True,
        'message': 'Syllabus cleared'
    })


# ===================
# STUDY MATERIALS ROUTES
# ===================

@app.route('/api/workspaces/<workspace_id>/study-materials', methods=['GET'])
@auth_required
def get_study_materials(workspace_id):
    """Get all generated study materials for a workspace."""
    # Verify workspace belongs to user
    workspace = db.get_workspace_by_id_and_user(workspace_id, request.user_id)
    if not workspace:
        return jsonify({'error': 'Workspace not found'}), 404

    materials = db.get_all_study_materials(workspace_id)

    return jsonify({
        'materials': [{
            'id': str(m['id']),
            'module_id': m['module_id'],
            'module_name': m['module_name'],
            'subtopic': m['subtopic'],
            'created_at': m['created_at'].isoformat() if m['created_at'] else None
        } for m in materials]
    })


@app.route('/api/workspaces/<workspace_id>/study-materials/<int:module_id>/<path:subtopic>', methods=['GET'])
@auth_required
def get_study_material_content(workspace_id, module_id, subtopic):
    """Get content for a specific subtopic."""
    # Verify workspace belongs to user
    workspace = db.get_workspace_by_id_and_user(workspace_id, request.user_id)
    if not workspace:
        return jsonify({'error': 'Workspace not found'}), 404

    # URL decode subtopic
    from urllib.parse import unquote
    subtopic = unquote(subtopic)

    material = db.get_study_material(workspace_id, module_id, subtopic)

    if not material:
        return jsonify({'error': 'Study material not found'}), 404

    return jsonify({
        'material': {
            'id': str(material['id']),
            'module_id': material['module_id'],
            'module_name': material['module_name'],
            'subtopic': material['subtopic'],
            'content': material['content'],
            'created_at': material['created_at'].isoformat() if material['created_at'] else None
        }
    })


@app.route('/api/workspaces/<workspace_id>/study-materials/generate', methods=['POST'])
@auth_required
def generate_study_material_route(workspace_id):
    """Generate study material for a subtopic."""
    # Verify workspace belongs to user
    workspace = db.get_workspace_by_id_and_user(workspace_id, request.user_id)
    if not workspace:
        return jsonify({'error': 'Workspace not found'}), 404

    data = request.json
    module_id = data.get('module_id')
    module_name = data.get('module_name')
    subtopic = data.get('subtopic')

    if not module_id or not module_name or not subtopic:
        return jsonify({'error': 'module_id, module_name, and subtopic are required'}), 400

    # Check if already exists
    existing = db.get_study_material(workspace_id, module_id, subtopic)
    if existing:
        return jsonify({
            'success': True,
            'material': {
                'id': str(existing['id']),
                'module_id': existing['module_id'],
                'module_name': existing['module_name'],
                'subtopic': existing['subtopic'],
                'content': existing['content'],
                'created_at': existing['created_at'].isoformat() if existing['created_at'] else None
            },
            'cached': True
        })

    # Try to get RAG context from uploaded documents
    rag_context = None
    try:
        documents = db.get_documents_by_workspace(workspace_id, request.user_id)
        if documents:
            # Use retrieval to get relevant chunks
            search_query = f"{module_name} {subtopic}"
            chunks = retrieve(search_query, request.user_id, workspace_id, top_k=5)
            if chunks:
                rag_context = "\n\n".join([c['text'] for c in chunks])
    except Exception as e:
        print(f"RAG retrieval error: {e}")

    # Generate study material
    result = generation.generate_study_material(module_name, subtopic, rag_context)

    if result.get('error'):
        return jsonify({'error': result['error']}), 500

    # Save to database
    saved = db.save_study_material(
        workspace_id,
        request.user_id,
        module_id,
        module_name,
        subtopic,
        result['content']
    )

    return jsonify({
        'success': True,
        'material': {
            'id': str(saved['id']),
            'module_id': saved['module_id'],
            'module_name': saved['module_name'],
            'subtopic': saved['subtopic'],
            'content': saved['content'],
            'created_at': saved['created_at'].isoformat() if saved['created_at'] else None
        },
        'cached': False
    })


@app.route('/api/workspaces/<workspace_id>/study-materials/generate-stream', methods=['POST'])
@auth_required
def generate_study_material_stream_route(workspace_id):
    """Generate study material for a subtopic with streaming."""
    # Capture user_id before generator (request context won't be available inside generator)
    user_id = request.user_id

    # Verify workspace belongs to user
    workspace = db.get_workspace_by_id_and_user(workspace_id, user_id)
    if not workspace:
        return jsonify({'error': 'Workspace not found'}), 404

    data = request.json
    module_id = data.get('module_id')
    module_name = data.get('module_name')
    subtopic = data.get('subtopic')
    regenerate = data.get('regenerate', False)

    if not module_id or not module_name or not subtopic:
        return jsonify({'error': 'module_id, module_name, and subtopic are required'}), 400

    # Check if already exists (unless regenerate is requested)
    if not regenerate:
        existing = db.get_study_material(workspace_id, module_id, subtopic)
        if existing:
            # Return existing content immediately
            def cached_response():
                yield f"data: {json.dumps({'type': 'cached', 'content': existing['content']})}\n\n"
            return Response(cached_response(), mimetype='text/event-stream')

    # Try to get RAG context from uploaded documents
    rag_context = None
    try:
        documents = db.get_documents_by_workspace(workspace_id, user_id)
        if documents:
            search_query = f"{module_name} {subtopic}"
            result = retrieve(search_query, user_id, workspace_id)
            if result and result.get('chunks'):
                rag_context = "\n\n".join(result['chunks'])
    except Exception as e:
        print(f"RAG retrieval error: {e}")

    def generate():
        full_content = ""
        try:
            for event in generation.generate_study_material_stream(module_name, subtopic, rag_context):
                if event['type'] == 'chunk':
                    full_content += event['content']
                    yield f"data: {json.dumps({'type': 'chunk', 'content': event['content']})}\n\n"
                elif event['type'] == 'done':
                    # Save to database (using captured user_id, not request.user_id)
                    try:
                        saved = db.save_study_material(
                            workspace_id,
                            user_id,
                            module_id,
                            module_name,
                            subtopic,
                            full_content
                        )
                        yield f"data: {json.dumps({'type': 'done', 'id': str(saved['id']) if saved else None})}\n\n"
                    except Exception as save_error:
                        yield f"data: {json.dumps({'type': 'error', 'message': f'Failed to save: {str(save_error)}'})}\n\n"
                elif event['type'] == 'error':
                    yield f"data: {json.dumps({'type': 'error', 'message': event['message']})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


# ===================
# CONVERSATION ROUTES
# ===================

@app.route('/api/conversations', methods=['GET'])
@auth_required
def list_conversations():
    """List conversations for a workspace."""
    workspace_id = request.args.get('workspace_id')
    if not workspace_id:
        return jsonify({'error': 'workspace_id required'}), 400

    conversations = db.list_conversations(request.user_id, workspace_id)
    return jsonify({
        'conversations': [{
            'id': str(c['id']),
            'title': c['title'],
            'updated_at': c['updated_at'].isoformat() if c['updated_at'] else None
        } for c in (conversations or [])]
    })


@app.route('/api/conversations', methods=['POST'])
@auth_required
def create_conversation():
    """Create a new conversation."""
    data = request.json
    workspace_id = data.get('workspace_id')

    if not workspace_id:
        return jsonify({'error': 'workspace_id required'}), 400

    conversation = db.create_conversation(request.user_id, workspace_id)
    return jsonify({
        'conversation': {
            'id': str(conversation['id']),
            'title': conversation['title']
        }
    })


@app.route('/api/conversations/<conversation_id>', methods=['GET'])
@auth_required
def get_conversation(conversation_id):
    """Get conversation with messages."""
    conversation = db.get_conversation(conversation_id, request.user_id)
    if not conversation:
        return jsonify({'error': 'Conversation not found'}), 404

    messages = db.get_messages(conversation_id)

    # Process messages - s3_key is used by frontend to construct proxy URL
    processed_messages = []
    for m in (messages or []):
        msg_data = {
            'id': str(m['id']),
            'role': m['role'],
            'content': m['content'],
            'sources': m['sources'] or [],
            'attachments': m['attachments'] or [],
            'thinking': m['thinking'],
            'created_at': m['created_at'].isoformat() if m['created_at'] else None
        }
        processed_messages.append(msg_data)

    return jsonify({
        'conversation': {
            'id': str(conversation['id']),
            'title': conversation['title']
        },
        'messages': processed_messages
    })


@app.route('/api/conversations/<conversation_id>', methods=['PUT'])
@auth_required
def update_conversation(conversation_id):
    """Update conversation title."""
    data = request.json
    title = data.get('title')

    if not title:
        return jsonify({'error': 'title required'}), 400

    conversation = db.update_conversation(conversation_id, request.user_id, title=title)
    if not conversation:
        return jsonify({'error': 'Conversation not found'}), 404

    return jsonify({'success': True})


@app.route('/api/conversations/<conversation_id>', methods=['DELETE'])
@auth_required
def delete_conversation_route(conversation_id):
    """Delete a conversation and its S3 images."""
    from services import storage

    # First, get S3 keys for any images in this conversation
    s3_keys = db.get_conversation_image_s3_keys(conversation_id)

    # Delete from database
    db.delete_conversation(conversation_id, request.user_id)

    # Delete images from S3
    for key in s3_keys:
        try:
            storage.delete_file(key)
        except Exception as e:
            print(f"Failed to delete S3 file {key}: {e}")

    return jsonify({'success': True, 'deleted_images': len(s3_keys)})


@app.route('/api/workspaces/<workspace_id>/conversations', methods=['DELETE'])
@auth_required
def delete_workspace_conversations(workspace_id):
    """Delete all conversations in a workspace and their S3 images."""
    from services import storage

    # Get all conversations in this workspace
    conversations = db.get_workspace_conversations(workspace_id, request.user_id)

    if not conversations:
        return jsonify({'success': True, 'deleted_count': 0, 'deleted_images': 0})

    # Collect all S3 keys from all conversations
    all_s3_keys = []
    for conv in conversations:
        s3_keys = db.get_conversation_image_s3_keys(conv['id'])
        all_s3_keys.extend(s3_keys)

    # Delete all conversations from database (cascade deletes messages)
    deleted_count = len(conversations)
    db.delete_workspace_conversations(workspace_id, request.user_id)

    # Delete all images from S3
    for key in all_s3_keys:
        try:
            storage.delete_file(key)
        except Exception as e:
            print(f"Failed to delete S3 file {key}: {e}")

    return jsonify({
        'success': True,
        'deleted_count': deleted_count,
        'deleted_images': len(all_s3_keys)
    })


@app.route('/api/workspaces/<workspace_id>/documents', methods=['DELETE'])
@auth_required
def delete_workspace_documents(workspace_id):
    """Delete all documents in a workspace from S3, ChromaDB, and database."""
    # Get all documents in this workspace
    documents = db.get_documents_by_workspace(workspace_id, request.user_id)

    if not documents:
        return jsonify({'success': True, 'deleted_count': 0})

    deleted_count = len(documents)

    # Delete from vector store (ChromaDB)
    for doc in documents:
        try:
            vector_store.delete_document(doc['filename'])
        except Exception as e:
            print(f"Error deleting vectors for doc {doc['id']}: {e}")

    # Invalidate BM25 cache
    invalidate_bm25_cache()

    # Delete from S3 - both raw and processed files
    try:
        for doc in documents:
            if doc.get('raw_storage_key'):
                storage.delete_file(doc['raw_storage_key'])
            if doc.get('processed_storage_key'):
                storage.delete_file(doc['processed_storage_key'])
    except Exception as e:
        print(f"Error deleting S3 files: {e}")

    # Delete from database
    db.delete_workspace_documents(workspace_id, request.user_id)

    return jsonify({
        'success': True,
        'deleted_count': deleted_count
    })


@app.route('/api/conversations/<conversation_id>/truncate', methods=['POST'])
@auth_required
def truncate_conversation(conversation_id):
    """Truncate conversation messages for regeneration."""
    conversation = db.get_conversation(conversation_id, request.user_id)
    if not conversation:
        return jsonify({'error': 'Conversation not found'}), 404

    data = request.json
    keep_until_user_message = data.get('keep_until_user_message', '')
    delete_user_message = data.get('delete_user_message', False)

    if not keep_until_user_message:
        return jsonify({'error': 'Missing keep_until_user_message'}), 400

    # Get all messages and find where to truncate
    messages = db.get_conversation_messages(conversation_id)

    # Find the last user message matching the content
    truncate_index = -1
    for i, msg in enumerate(messages):
        if msg['role'] == 'user' and msg['content'].strip() == keep_until_user_message.strip():
            truncate_index = i

    if truncate_index == -1:
        return jsonify({'error': 'User message not found'}), 404

    # If delete_user_message is True, also delete the user message itself (for edit)
    # Otherwise, keep up to and including the user message (for regenerate)
    if delete_user_message:
        messages_to_keep = messages[:truncate_index]  # Delete user message and everything after
    else:
        messages_to_keep = messages[:truncate_index + 1]  # Keep up to and including the user message

    db.truncate_conversation_messages(conversation_id, len(messages_to_keep))

    return jsonify({'success': True, 'kept_messages': len(messages_to_keep)})


@app.route('/api/conversations/<conversation_id>/chat', methods=['POST'])
@auth_required
def conversation_chat(conversation_id):
    """Send message in a conversation (streaming)."""
    conversation = db.get_conversation(conversation_id, request.user_id)
    if not conversation:
        return jsonify({'error': 'Conversation not found'}), 404

    data = request.json
    question = data.get('question', '').strip()
    use_search = data.get('use_search', False)
    use_thinking = data.get('use_thinking', False)
    use_rag = data.get('use_rag', False)
    is_regenerate = data.get('regenerate', False)

    if not question:
        return jsonify({'error': 'Question is required'}), 400

    workspace_id = str(conversation['workspace_id'])
    user_id = request.user_id

    # Check documents only if RAG is enabled
    if use_rag:
        user_docs = db.get_documents_by_workspace(workspace_id, user_id)
        if not user_docs:
            return jsonify({'error': 'No documents uploaded. Please upload documents or disable RAG.'}), 400

    # Save user message (skip if regenerating - user message already exists)
    if not is_regenerate:
        db.add_message(conversation_id, 'user', question)

    def generate():
        try:
            # Get recent messages for context
            recent = db.get_recent_messages(conversation_id, limit=10)
            chat_history = [{'question': m['content'], 'answer': ''} for m in recent if m['role'] == 'user']

            # Retrieve chunks only if RAG is enabled
            if use_rag:
                retrieval_result = retrieve(question, user_id=user_id, workspace_id=workspace_id)
                chunks = retrieval_result["chunks"]
                metadatas = retrieval_result["metadatas"]
            else:
                # No RAG - send directly to LLM without document context
                chunks = []
                metadatas = []

            full_answer = ""
            full_thinking = ""
            sources = []

            for event in generation.generate_answer_stream(
                query=question,
                chunks=chunks,
                metadatas=metadatas,
                chat_history=chat_history,
                use_search=use_search,
                use_thinking=use_thinking
            ):
                if event["type"] == "thinking":
                    full_thinking += event.get("content", "")
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

            # Save assistant message
            if full_answer:
                db.add_message(
                    conversation_id, 'assistant', full_answer,
                    sources=sources, thinking=full_thinking or None
                )

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/conversations/<conversation_id>/title', methods=['POST'])
@auth_required
def generate_conversation_title(conversation_id):
    """Generate title for conversation using AI based on Q&A context."""
    conversation = db.get_conversation(conversation_id, request.user_id)
    if not conversation:
        return jsonify({'error': 'Conversation not found'}), 404

    messages = db.get_messages(conversation_id, limit=2)
    if not messages:
        return jsonify({'error': 'No messages'}), 400

    # Get first user question and assistant answer
    user_msg = next((m['content'] for m in messages if m['role'] == 'user'), None)
    assistant_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), None)

    if not user_msg:
        return jsonify({'error': 'No user message'}), 400

    # Generate title using both question and answer for context
    try:
        context = f"Question: {user_msg[:150]}"
        if assistant_msg:
            context += f"\n\nAnswer summary: {assistant_msg[:300]}"

        prompt = f"Create a 3-5 word title for this conversation. Plain text only, no quotes, no punctuation at start/end.\n\n{context}"

        # Use Cerebras Llama 3.3 70B for title generation (ultra fast)
        from cerebras.cloud.sdk import Cerebras
        import os
        cerebras = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
        response = cerebras.chat.completions.create(
            model="llama-3.3-70b",
            messages=[{"role": "user", "content": prompt}]
        )
        title = response.choices[0].message.content

        if not title:
            title = user_msg[:40]
        # Clean up: remove quotes, markdown, extra punctuation
        title = title.strip().strip('"').strip("'").strip('*').strip(':').strip('-')
        title = title.replace('**', '').replace('*', '').replace('"', '').replace("'", "")
        title = title.split('\n')[0].strip()[:40]

        db.update_conversation(conversation_id, request.user_id, title=title)
        return jsonify({'title': title})
    except Exception as e:
        print(f"Error generating title: {e}")
        return jsonify({'title': user_msg[:40] + '...' if len(user_msg) > 40 else user_msg})


def get_user_from_token_or_query():
    """Get user_id from Authorization header or query param token."""
    from services.auth import decode_token
    token = None

    # Try Authorization header first
    if 'Authorization' in request.headers:
        auth_header = request.headers['Authorization']
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]

    # Fallback to query parameter (for direct browser requests)
    if not token:
        token = request.args.get('token')

    if not token:
        return None

    try:
        payload = decode_token(token)
        return payload['user_id']
    except:
        return None


@app.route('/api/images/<path:s3_key>', methods=['GET'])
def serve_image(s3_key):
    """Serve an image from S3 (proxy to avoid CORS issues)."""
    import io
    from flask import send_file

    user_id = get_user_from_token_or_query()
    if not user_id:
        return jsonify({'error': 'Authentication required'}), 401

    # Security check: ensure the s3_key belongs to this user
    if not s3_key.startswith(f"{user_id}/"):
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        # Download from S3 to memory
        file_obj = io.BytesIO()
        storage.download_fileobj(s3_key, file_obj)
        file_obj.seek(0)

        # Determine mime type from extension
        mime_type = 'image/png' if s3_key.endswith('.png') else 'image/jpeg'

        return send_file(file_obj, mimetype=mime_type)
    except Exception as e:
        print(f"Error serving image {s3_key}: {e}")
        return jsonify({'error': 'Image not found'}), 404


@app.route('/api/images/<path:s3_key>/download', methods=['GET'])
def download_image(s3_key):
    """Download an image from S3 with Content-Disposition header."""
    import io
    from flask import send_file

    user_id = get_user_from_token_or_query()
    if not user_id:
        return jsonify({'error': 'Authentication required'}), 401

    # Security check: ensure the s3_key belongs to this user
    if not s3_key.startswith(f"{user_id}/"):
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        # Download from S3 to memory
        file_obj = io.BytesIO()
        storage.download_fileobj(s3_key, file_obj)
        file_obj.seek(0)

        # Determine mime type and filename from key
        mime_type = 'image/png' if s3_key.endswith('.png') else 'image/jpeg'
        filename = s3_key.split('/')[-1]

        return send_file(
            file_obj,
            mimetype=mime_type,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        print(f"Error downloading image {s3_key}: {e}")
        return jsonify({'error': 'Image not found'}), 404


@app.route('/api/conversations/<conversation_id>/image', methods=['POST'])
@auth_required
def conversation_image(conversation_id):
    """Generate image and save to conversation."""
    conversation = db.get_conversation(conversation_id, request.user_id)
    if not conversation:
        return jsonify({'error': 'Conversation not found'}), 404

    data = request.json
    prompt = data.get('prompt', '').strip()
    aspect_ratio = data.get('aspect_ratio', '1:1')

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    # Save user message
    db.add_message(conversation_id, 'user', prompt)

    # Check if current model supports image generation
    current_model = llm.get_current_model()
    if current_model.get('type') != 'image':
        return jsonify({'error': f"Current model '{current_model['name']}' does not support image generation"}), 400

    try:
        result = llm.generate_image(prompt, aspect_ratio=aspect_ratio)

        # Upload image to S3 instead of storing base64 in DB
        import base64
        import uuid
        from services import storage

        image_bytes = base64.b64decode(result['image_base64'])
        image_id = f"img_{uuid.uuid4().hex[:12]}"
        extension = 'png' if 'png' in result['mime_type'] else 'jpg'

        # Generate S3 key: user_id/workspace_id/generatedimages/image_id.ext
        s3_key = storage.generate_image_key(
            request.user_id,
            conversation['workspace_id'],
            image_id,
            extension
        )

        # Upload to S3
        storage.upload_image_bytes(s3_key, image_bytes, result['mime_type'])

        # Save assistant message with S3 reference (not base64)
        db.add_message(
            conversation_id, 'assistant', f'Generated image: {prompt}',
            attachments=[{
                'type': 'image',
                's3_key': s3_key,
                'mime_type': result['mime_type'],
                'prompt': prompt,
                'aspect_ratio': aspect_ratio
            }]
        )

        return jsonify({
            'success': True,
            's3_key': s3_key,
            'mime_type': result['mime_type'],
            'prompt': prompt,
            'aspect_ratio': aspect_ratio
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
            "original_filename": doc['original_filename'],
            "file_type": doc.get('file_type', 'pdf'),
            "processing_method": doc.get('processing_method', 'direct'),
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
    Upload and process a document (PDF, Word, PPT, Excel, Image, Audio).

    Expects:
        - file: Document file (multipart/form-data)
        - workspace_id: Workspace to upload to
        - use_ocr: (optional) Force OCR processing (true/false)
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    workspace_id = request.form.get('workspace_id')
    use_ocr_str = request.form.get('use_ocr')

    use_ocr = None
    if use_ocr_str == 'true':
        use_ocr = True
    elif use_ocr_str == 'false':
        use_ocr = False

    if not workspace_id:
        return jsonify({"error": "Workspace ID is required"}), 400

    workspace = db.get_workspace_by_id_and_user(workspace_id, request.user_id)
    if not workspace:
        return jsonify({"error": "Workspace not found"}), 404

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        supported = file_types.get_supported_formats()
        return jsonify({"error": f"Unsupported file type. Supported: {list(supported.keys())}"}), 400

    temp_file_path = None

    try:
        doc_uuid = str(uuid.uuid4())
        original_filename = file.filename
        safe_filename = secure_filename(file.filename)
        extension = file_types.get_extension(original_filename)
        file_type = file_types.get_file_type(original_filename)
        mime_type = file_types.get_mime_type(original_filename)

        temp_file_path = os.path.join(UPLOADS_PATH, f"temp_{doc_uuid}{extension}")
        file.save(temp_file_path)
        print(f"[UPLOAD] File saved to: {temp_file_path}")

        file_size = os.path.getsize(temp_file_path)
        content_hash = file_types.calculate_file_hash(temp_file_path)
        print(f"[UPLOAD] File size: {file_size}, hash: {content_hash[:16]}..., use_ocr: {use_ocr}")

        print(f"[UPLOAD] Starting file processing...")
        result = file_types.process_file(temp_file_path, use_ocr=use_ocr)
        print(f"[UPLOAD] Processing complete. Method: {result.get('processing_method')}, text length: {len(result.get('text', ''))}")

        raw_key = storage.generate_raw_key(request.user_id, workspace_id, doc_uuid, extension)
        storage.upload_file(raw_key, temp_file_path, mime_type)

        processed_key = storage.generate_processed_key(request.user_id, workspace_id, doc_uuid)
        storage.upload_text(processed_key, result["text"])

        chunks = chunking.chunk_text(
            text=result["text"],
            document_id=doc_uuid,
            document_name=original_filename,
            document_type=file_type
        )

        if not chunks:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return jsonify({"error": "Could not extract text from file"}), 400

        for chunk in chunks:
            chunk["metadata"]["user_id"] = request.user_id
            chunk["metadata"]["workspace_id"] = workspace_id

        chunk_texts = [c["text"] for c in chunks]
        embeddings = embedding.embed_texts(chunk_texts)
        metadatas = [c["metadata"] for c in chunks]

        vector_store.add_chunks(
            chunks=chunk_texts,
            embeddings=embeddings,
            metadatas=metadatas,
            document_id=doc_uuid
        )

        invalidate_bm25_cache()

        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        temp_file_path = None

        doc = db.create_document(
            workspace_id=workspace_id,
            user_id=request.user_id,
            filename=doc_uuid,
            original_filename=original_filename,
            file_type=file_type,
            file_size_bytes=file_size,
            storage_key=raw_key,
            storage_bucket=storage.BUCKET_NAME,
            num_pages=result.get("num_pages"),
            num_chunks=len(chunks),
            status='processed',
            raw_storage_key=raw_key,
            processed_storage_key=processed_key,
            processing_method=result.get("processing_method", "direct"),
            content_hash=content_hash,
            extracted_text_length=len(result["text"]),
            mime_type=mime_type,
            duration_seconds=result.get("duration_seconds")
        )

        return jsonify({
            "success": True,
            "message": f"Document '{original_filename}' processed successfully!",
            "document": {
                "id": str(doc['id']),
                "filename": original_filename,
                "file_type": file_type,
                "pages": result.get("num_pages"),
                "chunks": len(chunks),
                "processing_method": result.get("processing_method", "direct"),
                "storage": "s3"
            }
        })

    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return jsonify({"error": str(e)}), 500


@app.route('/api/upload/analyze', methods=['POST'])
@auth_required
def analyze_upload():
    """Pre-analyze files to determine OCR recommendations."""
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist('files')
    results = []

    for file in files[:3]:
        if not file.filename:
            continue

        if not allowed_file(file.filename):
            results.append({
                "name": file.filename,
                "supported": False,
                "error": "Unsupported file type"
            })
            continue

        temp_path = os.path.join(UPLOADS_PATH, f"analyze_{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}")
        try:
            file.save(temp_path)
            analysis = file_types.analyze_file(temp_path)

            results.append({
                "name": file.filename,
                "supported": True,
                "size": os.path.getsize(temp_path),
                "file_type": analysis["file_type"],
                "ocr_available": analysis["ocr_available"],
                "ocr_recommended": analysis["ocr_recommended"],
                "ocr_reason": analysis["ocr_reason"],
                "num_pages": analysis["num_pages"]
            })
        except Exception as e:
            results.append({
                "name": file.filename,
                "supported": False,
                "error": str(e)
            })
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return jsonify({"files": results})


@app.route('/api/upload/formats', methods=['GET'])
def get_supported_formats():
    """Get list of supported file formats."""
    return jsonify(file_types.get_supported_formats())


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

        # Delete from S3 - both raw and processed files
        if doc.get('raw_storage_key'):
            storage.delete_file(doc['raw_storage_key'])
        if doc.get('processed_storage_key'):
            storage.delete_file(doc['processed_storage_key'])

        # Delete from database (returns deleted doc for confirmation)
        db.delete_document(doc_id, request.user_id)

        return jsonify({
            "success": True,
            "message": f"Document '{doc.get('original_filename', doc_id)}' deleted"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/documents/<doc_id>/url/<file_type>', methods=['GET'])
@auth_required
def get_document_url(doc_id, file_type):
    """Get presigned URL for document file (raw or processed)."""
    try:
        doc = db.get_document_by_id_and_user(doc_id, request.user_id)

        if not doc:
            return jsonify({"error": "Document not found"}), 404

        if file_type == 'raw':
            key = doc.get('raw_storage_key')
        elif file_type == 'processed':
            key = doc.get('processed_storage_key')
        else:
            return jsonify({"error": "Invalid file type. Use 'raw' or 'processed'"}), 400

        if not key:
            return jsonify({"error": f"No {file_type} file available"}), 404

        # Generate presigned URL with 1 hour expiry
        url = storage.get_download_url(key, expires_in=3600)

        return jsonify({
            "url": url,
            "expires_in": 3600
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/documents/<doc_id>/content', methods=['GET'])
@auth_required
def get_document_content(doc_id):
    """Get document metadata and processed content."""
    try:
        doc = db.get_document_by_id_and_user(doc_id, request.user_id)

        if not doc:
            return jsonify({"error": "Document not found"}), 404

        # Get processed content from S3
        processed_key = doc.get('processed_storage_key')
        if not processed_key:
            return jsonify({"error": "No processed content available"}), 404

        content = storage.download_text(processed_key)

        return jsonify({
            "id": str(doc['id']),
            "filename": doc['original_filename'],
            "file_type": doc.get('file_type', 'pdf'),
            "num_pages": doc.get('num_pages'),
            "processing_method": doc.get('processing_method', 'direct'),
            "content": content
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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


@app.route('/api/stats', methods=['GET'])
@auth_required
def get_stats():
    """Get statistics for user's workspace."""
    workspace_id = request.args.get('workspace_id')

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
        "total_size_mb": round(total_size / (1024 * 1024), 2)
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
