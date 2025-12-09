"""
S3/R2 Storage Service for LearnForge.

Handles file uploads, downloads, and deletions using presigned URLs.
Supports both AWS S3 and Cloudflare R2 (S3-compatible).

File hierarchy: {user_id}/{workspace_id}/{document_id}_{filename}
"""

import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from typing import Optional
import uuid

load_dotenv()


class StorageService:
    """S3/R2 storage service with presigned URL support."""

    def __init__(self):
        self.endpoint_url = os.getenv('S3_ENDPOINT_URL')
        self.access_key = os.getenv('S3_ACCESS_KEY')
        self.secret_key = os.getenv('S3_SECRET_KEY')
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        self.region = os.getenv('S3_REGION', 'auto')

        if not all([self.endpoint_url, self.access_key, self.secret_key, self.bucket_name]):
            raise ValueError(
                "S3 not configured. Set S3_ENDPOINT_URL, S3_ACCESS_KEY, "
                "S3_SECRET_KEY, and S3_BUCKET_NAME in .env"
            )

        # Create S3 client
        self.client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
            config=Config(
                signature_version='s3v4',
                s3={'addressing_style': 'path'}
            )
        )

    def generate_key(self, user_id: str, workspace_id: str, filename: str, doc_id: str = None) -> str:
        """
        Generate storage key with hierarchy.

        Format: {user_id}/{workspace_id}/{doc_id}_{filename}
        """
        if not doc_id:
            doc_id = f"doc_{uuid.uuid4().hex[:12]}"

        # Sanitize filename
        safe_filename = "".join(c for c in filename if c.isalnum() or c in '._-').strip()
        if not safe_filename:
            safe_filename = "document.pdf"

        return f"{user_id}/{workspace_id}/{doc_id}_{safe_filename}"

    def get_upload_presigned_url(
        self,
        key: str,
        content_type: str = 'application/pdf',
        expires_in: int = 3600
    ) -> dict:
        """
        Generate presigned URL for direct upload to S3.

        Args:
            key: Storage key (path)
            content_type: MIME type of file
            expires_in: URL expiration in seconds

        Returns:
            Dict with 'url' and 'fields' for form upload, or 'url' for PUT
        """
        try:
            # Generate presigned POST for form-based upload
            response = self.client.generate_presigned_post(
                Bucket=self.bucket_name,
                Key=key,
                Fields={'Content-Type': content_type},
                Conditions=[
                    {'Content-Type': content_type},
                    ['content-length-range', 1, 104857600]  # 1 byte to 100MB
                ],
                ExpiresIn=expires_in
            )
            return {
                'method': 'POST',
                'url': response['url'],
                'fields': response['fields']
            }
        except ClientError as e:
            print(f"[STORAGE ERROR] Failed to generate upload URL: {e}")
            raise

    def get_download_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """
        Generate presigned URL for downloading a file.

        Args:
            key: Storage key (path)
            expires_in: URL expiration in seconds

        Returns:
            Presigned download URL
        """
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': key},
                ExpiresIn=expires_in
            )
            return url
        except ClientError as e:
            print(f"[STORAGE ERROR] Failed to generate download URL: {e}")
            raise

    def upload_file(self, key: str, file_path: str, content_type: str = 'application/pdf') -> bool:
        """
        Upload a local file to S3.

        Args:
            key: Storage key (path)
            file_path: Local file path
            content_type: MIME type

        Returns:
            True if successful
        """
        try:
            self.client.upload_file(
                file_path,
                self.bucket_name,
                key,
                ExtraArgs={'ContentType': content_type}
            )
            return True
        except ClientError as e:
            print(f"[STORAGE ERROR] Failed to upload file: {e}")
            raise

    def upload_fileobj(self, key: str, file_obj, content_type: str = 'application/pdf') -> bool:
        """
        Upload a file object to S3.

        Args:
            key: Storage key (path)
            file_obj: File-like object
            content_type: MIME type

        Returns:
            True if successful
        """
        try:
            self.client.upload_fileobj(
                file_obj,
                self.bucket_name,
                key,
                ExtraArgs={'ContentType': content_type}
            )
            return True
        except ClientError as e:
            print(f"[STORAGE ERROR] Failed to upload file object: {e}")
            raise

    def download_file(self, key: str, file_path: str) -> bool:
        """
        Download a file from S3 to local path.

        Args:
            key: Storage key (path)
            file_path: Local destination path

        Returns:
            True if successful
        """
        try:
            self.client.download_file(self.bucket_name, key, file_path)
            return True
        except ClientError as e:
            print(f"[STORAGE ERROR] Failed to download file: {e}")
            raise

    def download_fileobj(self, key: str, file_obj) -> bool:
        """
        Download a file from S3 to a file object.

        Args:
            key: Storage key (path)
            file_obj: File-like object to write to

        Returns:
            True if successful
        """
        try:
            self.client.download_fileobj(self.bucket_name, key, file_obj)
            return True
        except ClientError as e:
            print(f"[STORAGE ERROR] Failed to download file object: {e}")
            raise

    def delete_file(self, key: str) -> bool:
        """
        Delete a file from S3.

        Args:
            key: Storage key (path)

        Returns:
            True if successful
        """
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as e:
            print(f"[STORAGE ERROR] Failed to delete file: {e}")
            raise

    def delete_folder(self, prefix: str) -> int:
        """
        Delete all files under a prefix (folder).

        Args:
            prefix: Folder prefix (e.g., "user_id/workspace_id/")

        Returns:
            Number of files deleted
        """
        try:
            # List all objects with prefix
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )

            if 'Contents' not in response:
                return 0

            # Delete all objects
            objects = [{'Key': obj['Key']} for obj in response['Contents']]

            if objects:
                self.client.delete_objects(
                    Bucket=self.bucket_name,
                    Delete={'Objects': objects}
                )

            return len(objects)
        except ClientError as e:
            print(f"[STORAGE ERROR] Failed to delete folder: {e}")
            raise

    def file_exists(self, key: str) -> bool:
        """
        Check if a file exists in S3.

        Args:
            key: Storage key (path)

        Returns:
            True if file exists
        """
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise

    def get_file_size(self, key: str) -> Optional[int]:
        """
        Get file size in bytes.

        Args:
            key: Storage key (path)

        Returns:
            File size in bytes, or None if not found
        """
        try:
            response = self.client.head_object(Bucket=self.bucket_name, Key=key)
            return response['ContentLength']
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return None
            raise
