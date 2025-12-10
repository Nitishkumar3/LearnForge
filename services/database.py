"""
Database service using raw SQL queries.
Simple approach - one reusable connection, one query executor.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from dotenv import load_dotenv

load_dotenv()

# Reusable connection
conn = None


def get_connection():
    """Get or create database connection."""
    global conn
    if conn is None or conn.closed:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT", 5432),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD")
        )
    return conn


def execute_query(query, params=None, fetch_one=False, fetch_all=False):
    """
    Execute SQL query and return results.

    Args:
        query: SQL query string
        params: Tuple of parameters
        fetch_one: Return single row
        fetch_all: Return all rows

    Returns:
        - For SELECT: dict (fetch_one) or list of dicts (fetch_all)
        - For INSERT/UPDATE/DELETE with RETURNING: the returned row(s)
        - For other queries: True on success
    """
    connection = get_connection()
    cur = connection.cursor(cursor_factory=RealDictCursor)

    try:
        cur.execute(query, params)

        if fetch_one:
            result = cur.fetchone()
        elif fetch_all:
            result = cur.fetchall()
        else:
            result = True

        connection.commit()
        return result

    except Exception as e:
        connection.rollback()
        raise e

    finally:
        cur.close()


def init_db():
    """Initialize database from schema.sql."""
    schema_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "schema.sql"
    )

    with open(schema_path, 'r') as f:
        schema = f.read()

    connection = get_connection()
    cur = connection.cursor()

    try:
        cur.execute(schema)
        connection.commit()
    finally:
        cur.close()


# =============================================
# USER QUERIES
# =============================================

def create_user(name, email, password_hash, verification_token=None):
    """Insert new user and return the created user."""
    return execute_query(
        """INSERT INTO users (name, email, password_hash, verification_token)
           VALUES (%s, %s, %s, %s) RETURNING *""",
        (name, email.lower(), password_hash, verification_token),
        fetch_one=True
    )


def get_user_by_id(user_id):
    """Get user by ID."""
    return execute_query(
        "SELECT * FROM users WHERE id = %s",
        (user_id,),
        fetch_one=True
    )


def get_user_by_email(email):
    """Get user by email."""
    return execute_query(
        "SELECT * FROM users WHERE email = %s",
        (email.lower(),),
        fetch_one=True
    )


def get_user_by_verification_token(token):
    """Get user by verification token."""
    return execute_query(
        "SELECT * FROM users WHERE verification_token = %s",
        (token,),
        fetch_one=True
    )


def get_user_by_reset_token(token):
    """Get user by reset token."""
    return execute_query(
        "SELECT * FROM users WHERE reset_token = %s",
        (token,),
        fetch_one=True
    )


def verify_user_email(user_id):
    """Mark user email as verified."""
    return execute_query(
        "UPDATE users SET email_verified = TRUE, verification_token = NULL WHERE id = %s",
        (user_id,)
    )


def set_reset_token(user_id, token, expires):
    """Set password reset token."""
    return execute_query(
        "UPDATE users SET reset_token = %s, reset_token_expires = %s WHERE id = %s",
        (token, expires, user_id)
    )


def update_password(user_id, password_hash):
    """Update user password and clear reset token."""
    return execute_query(
        "UPDATE users SET password_hash = %s, reset_token = NULL, reset_token_expires = NULL WHERE id = %s",
        (password_hash, user_id)
    )


# =============================================
# WORKSPACE QUERIES
# =============================================

def create_workspace(user_id, name, description=None):
    """Insert new workspace and return it."""
    return execute_query(
        "INSERT INTO workspaces (user_id, name, description) VALUES (%s, %s, %s) RETURNING *",
        (user_id, name, description),
        fetch_one=True
    )


def get_workspace_by_id(workspace_id):
    """Get workspace by ID."""
    return execute_query(
        "SELECT * FROM workspaces WHERE id = %s",
        (workspace_id,),
        fetch_one=True
    )


def get_workspace_by_id_and_user(workspace_id, user_id):
    """Get workspace by ID, ensuring it belongs to user."""
    return execute_query(
        "SELECT * FROM workspaces WHERE id = %s AND user_id = %s",
        (workspace_id, user_id),
        fetch_one=True
    )


def get_workspaces_by_user(user_id):
    """Get all workspaces for a user."""
    return execute_query(
        "SELECT * FROM workspaces WHERE user_id = %s ORDER BY created_at DESC",
        (user_id,),
        fetch_all=True
    )


def update_workspace(workspace_id, user_id, name=None, description=None):
    """Update workspace and return updated record."""
    if name and description:
        return execute_query(
            "UPDATE workspaces SET name = %s, description = %s WHERE id = %s AND user_id = %s RETURNING *",
            (name, description, workspace_id, user_id),
            fetch_one=True
        )
    elif name:
        return execute_query(
            "UPDATE workspaces SET name = %s WHERE id = %s AND user_id = %s RETURNING *",
            (name, workspace_id, user_id),
            fetch_one=True
        )
    elif description:
        return execute_query(
            "UPDATE workspaces SET description = %s WHERE id = %s AND user_id = %s RETURNING *",
            (description, workspace_id, user_id),
            fetch_one=True
        )
    return get_workspace_by_id(workspace_id)


def delete_workspace(workspace_id, user_id):
    """Delete workspace (cascades to documents)."""
    return execute_query(
        "DELETE FROM workspaces WHERE id = %s AND user_id = %s",
        (workspace_id, user_id)
    )


# =============================================
# DOCUMENT QUERIES
# =============================================

def create_document(workspace_id, user_id, filename, original_filename,
                    file_type, file_size_bytes, storage_key, storage_bucket,
                    num_pages=None, num_chunks=None, status='pending'):
    """Insert new document and return it."""
    return execute_query(
        """INSERT INTO documents
           (workspace_id, user_id, filename, original_filename,
            file_type, file_size_bytes, storage_key, storage_bucket,
            num_pages, num_chunks, status)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
           RETURNING *""",
        (workspace_id, user_id, filename, original_filename,
         file_type, file_size_bytes, storage_key, storage_bucket,
         num_pages, num_chunks, status),
        fetch_one=True
    )


def get_document_by_id(doc_id):
    """Get document by ID."""
    return execute_query(
        "SELECT * FROM documents WHERE id = %s",
        (doc_id,),
        fetch_one=True
    )


def get_document_by_id_and_user(doc_id, user_id):
    """Get document by ID, ensuring it belongs to user."""
    return execute_query(
        "SELECT * FROM documents WHERE id = %s AND user_id = %s",
        (doc_id, user_id),
        fetch_one=True
    )


def get_documents_by_workspace(workspace_id, user_id):
    """Get all documents in a workspace."""
    return execute_query(
        """SELECT * FROM documents
           WHERE workspace_id = %s AND user_id = %s
           ORDER BY created_at DESC""",
        (workspace_id, user_id),
        fetch_all=True
    )


def get_documents_by_user(user_id):
    """Get all documents for a user."""
    return execute_query(
        "SELECT * FROM documents WHERE user_id = %s ORDER BY created_at DESC",
        (user_id,),
        fetch_all=True
    )


def update_document_status(doc_id, status, num_pages=None, num_chunks=None,
                           error_message=None):
    """Update document processing status."""
    if status == 'processed':
        return execute_query(
            """UPDATE documents
               SET status = %s, num_pages = %s, num_chunks = %s, processed_at = NOW()
               WHERE id = %s""",
            (status, num_pages, num_chunks, doc_id)
        )
    elif status == 'failed':
        return execute_query(
            "UPDATE documents SET status = %s, error_message = %s WHERE id = %s",
            (status, error_message, doc_id)
        )
    else:
        return execute_query(
            "UPDATE documents SET status = %s WHERE id = %s",
            (status, doc_id)
        )


def delete_document(doc_id, user_id):
    """Delete document, return deleted doc info for cleanup."""
    return execute_query(
        "DELETE FROM documents WHERE id = %s AND user_id = %s RETURNING *",
        (doc_id, user_id),
        fetch_one=True
    )


def get_document_count_by_workspace(workspace_id):
    """Get document count for a workspace."""
    result = execute_query(
        "SELECT COUNT(*) as count FROM documents WHERE workspace_id = %s",
        (workspace_id,),
        fetch_one=True
    )
    return result['count'] if result else 0


# =============================================
# CONVERSATION QUERIES
# =============================================

def create_conversation(user_id, workspace_id, title="New Chat"):
    """Create a new conversation."""
    return execute_query(
        """INSERT INTO conversations (user_id, workspace_id, title)
           VALUES (%s, %s, %s)
           RETURNING *""",
        (user_id, workspace_id, title),
        fetch_one=True
    )


def get_conversation(conversation_id, user_id):
    """Get conversation by ID (with ownership check)."""
    return execute_query(
        "SELECT * FROM conversations WHERE id = %s AND user_id = %s",
        (conversation_id, user_id),
        fetch_one=True
    )


def list_conversations(user_id, workspace_id, limit=50):
    """List conversations for a workspace, ordered by most recent."""
    return execute_query(
        """SELECT c.*,
                  (SELECT content FROM messages
                   WHERE conversation_id = c.id
                   ORDER BY created_at DESC LIMIT 1) as last_message
           FROM conversations c
           WHERE c.user_id = %s AND c.workspace_id = %s AND c.is_archived = FALSE
           ORDER BY c.updated_at DESC
           LIMIT %s""",
        (user_id, workspace_id, limit),
        fetch_all=True
    )


def update_conversation(conversation_id, user_id, title=None, is_archived=None):
    """Update conversation title or archive status."""
    updates = []
    params = []

    if title is not None:
        updates.append("title = %s")
        params.append(title)
    if is_archived is not None:
        updates.append("is_archived = %s")
        params.append(is_archived)

    if not updates:
        return get_conversation(conversation_id, user_id)

    params.extend([conversation_id, user_id])

    return execute_query(
        f"""UPDATE conversations
            SET {', '.join(updates)}, updated_at = NOW()
            WHERE id = %s AND user_id = %s
            RETURNING *""",
        params,
        fetch_one=True
    )


def delete_conversation(conversation_id, user_id):
    """Delete conversation and all messages (cascade)."""
    return execute_query(
        "DELETE FROM conversations WHERE id = %s AND user_id = %s",
        (conversation_id, user_id)
    )


def get_conversation_image_s3_keys(conversation_id):
    """Get all S3 keys for images in a conversation's messages."""
    messages = execute_query(
        "SELECT attachments FROM messages WHERE conversation_id = %s AND attachments IS NOT NULL",
        (conversation_id,),
        fetch_all=True
    ) or []

    s3_keys = []
    for msg in messages:
        attachments = msg.get('attachments') or []
        for att in attachments:
            if att.get('type') == 'image' and att.get('s3_key'):
                s3_keys.append(att['s3_key'])
    return s3_keys


def get_workspace_conversations(workspace_id, user_id):
    """Get all conversations in a workspace for a user."""
    return execute_query(
        "SELECT id FROM conversations WHERE workspace_id = %s AND user_id = %s",
        (workspace_id, user_id),
        fetch_all=True
    ) or []


def delete_workspace_conversations(workspace_id, user_id):
    """Delete all conversations in a workspace. Returns count of deleted conversations."""
    result = execute_query(
        "DELETE FROM conversations WHERE workspace_id = %s AND user_id = %s",
        (workspace_id, user_id)
    )
    return result if result else 0


def delete_workspace_documents(workspace_id, user_id):
    """Delete all documents in a workspace."""
    return execute_query(
        "DELETE FROM documents WHERE workspace_id = %s AND user_id = %s",
        (workspace_id, user_id)
    )


# =============================================
# MESSAGE QUERIES
# =============================================

def add_message(conversation_id, role, content, sources=None, attachments=None,
                thinking=None, tokens_used=None, model_used=None):
    """Add a message to a conversation."""
    return execute_query(
        """INSERT INTO messages
           (conversation_id, role, content, sources, attachments,
            thinking, tokens_used, model_used)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
           RETURNING *""",
        (conversation_id, role, content,
         Json(sources or []), Json(attachments or []),
         thinking, tokens_used, model_used),
        fetch_one=True
    )


def get_messages(conversation_id, limit=100):
    """Get all messages in a conversation."""
    return execute_query(
        """SELECT * FROM messages
           WHERE conversation_id = %s
           ORDER BY created_at ASC
           LIMIT %s""",
        (conversation_id, limit),
        fetch_all=True
    )


def get_recent_messages(conversation_id, limit=10):
    """Get recent messages for context (used in RAG)."""
    messages = execute_query(
        """SELECT role, content FROM messages
           WHERE conversation_id = %s
           ORDER BY created_at DESC
           LIMIT %s""",
        (conversation_id, limit),
        fetch_all=True
    )
    # Reverse to get chronological order
    return messages[::-1] if messages else []


def get_conversation_messages(conversation_id):
    """Get all messages in a conversation ordered by creation time."""
    return execute_query(
        """SELECT id, role, content, created_at FROM messages
           WHERE conversation_id = %s
           ORDER BY created_at ASC""",
        (conversation_id,),
        fetch_all=True
    ) or []


def truncate_conversation_messages(conversation_id, keep_count):
    """Delete messages after a certain point in the conversation."""
    # Get all message IDs in order
    messages = execute_query(
        """SELECT id FROM messages
           WHERE conversation_id = %s
           ORDER BY created_at ASC""",
        (conversation_id,),
        fetch_all=True
    )

    if not messages or keep_count >= len(messages):
        return 0

    # Get IDs of messages to delete
    message_ids_to_delete = [msg['id'] for msg in messages[keep_count:]]

    if message_ids_to_delete:
        placeholders = ','.join(['%s'] * len(message_ids_to_delete))
        execute_query(
            f"DELETE FROM messages WHERE id IN ({placeholders})",
            tuple(message_ids_to_delete)
        )

    return len(message_ids_to_delete)
