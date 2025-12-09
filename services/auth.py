"""
Authentication service using JWT and bcrypt.
Simple approach - no complex session management.
"""

import jwt
import bcrypt
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, current_app
import secrets


class AuthService:
    def __init__(self, secret_key, token_expiry_hours=24 * 7):
        self.secret_key = secret_key
        self.token_expiry_hours = token_expiry_hours

    def hash_password(self, password):
        """Hash password using bcrypt."""
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    def verify_password(self, password, password_hash):
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode(), password_hash.encode())

    def generate_token(self, user_id):
        """Generate JWT access token."""
        payload = {
            'user_id': str(user_id),
            'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')

    def decode_token(self, token):
        """Decode and validate JWT token."""
        try:
            return jwt.decode(token, self.secret_key, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")

    def generate_verification_token(self):
        """Generate email verification token."""
        return secrets.token_urlsafe(32)

    def generate_reset_token(self):
        """Generate password reset token."""
        return secrets.token_urlsafe(32)


def auth_required(f):
    """Decorator to require authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        # Get token from header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]

        if not token:
            return jsonify({'error': 'Authentication required'}), 401

        try:
            auth_service = current_app.config['AUTH_SERVICE']
            payload = auth_service.decode_token(token)
            request.user_id = payload['user_id']
        except ValueError as e:
            return jsonify({'error': str(e)}), 401

        return f(*args, **kwargs)

    return decorated
