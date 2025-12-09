"""S3/R2 storage utilities."""

import os
import uuid
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
ACCESS_KEY = os.getenv('S3_ACCESS_KEY')
SECRET_KEY = os.getenv('S3_SECRET_KEY')
BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
REGION = os.getenv('S3_REGION', 'auto')

client = None


def get_client():
    global client
    if client is None:
        client = boto3.client(
            's3',
            endpoint_url=ENDPOINT_URL,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            region_name=REGION,
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )
    return client


def generate_key(user_id, workspace_id, filename, doc_id=None):
    if not doc_id:
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    safe_filename = "".join(c for c in filename if c.isalnum() or c in '._-').strip()
    if not safe_filename:
        safe_filename = "document.pdf"
    return f"{user_id}/{workspace_id}/{doc_id}_{safe_filename}"


def get_upload_url(key, content_type='application/pdf', expires_in=3600):
    response = get_client().generate_presigned_post(
        Bucket=BUCKET_NAME,
        Key=key,
        Fields={'Content-Type': content_type},
        Conditions=[{'Content-Type': content_type}, ['content-length-range', 1, 104857600]],
        ExpiresIn=expires_in
    )
    return {'method': 'POST', 'url': response['url'], 'fields': response['fields']}


def get_download_url(key, expires_in=3600):
    return get_client().generate_presigned_url(
        'get_object',
        Params={'Bucket': BUCKET_NAME, 'Key': key},
        ExpiresIn=expires_in
    )


def upload_file(key, file_path, content_type='application/pdf'):
    get_client().upload_file(file_path, BUCKET_NAME, key, ExtraArgs={'ContentType': content_type})
    return True


def upload_fileobj(key, file_obj, content_type='application/pdf'):
    get_client().upload_fileobj(file_obj, BUCKET_NAME, key, ExtraArgs={'ContentType': content_type})
    return True


def download_file(key, file_path):
    get_client().download_file(BUCKET_NAME, key, file_path)
    return True


def download_fileobj(key, file_obj):
    get_client().download_fileobj(BUCKET_NAME, key, file_obj)
    return True


def delete_file(key):
    get_client().delete_object(Bucket=BUCKET_NAME, Key=key)
    return True


def delete_folder(prefix):
    response = get_client().list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    if 'Contents' not in response:
        return 0
    objects = [{'Key': obj['Key']} for obj in response['Contents']]
    if objects:
        get_client().delete_objects(Bucket=BUCKET_NAME, Delete={'Objects': objects})
    return len(objects)


def file_exists(key):
    try:
        get_client().head_object(Bucket=BUCKET_NAME, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise


def get_file_size(key):
    try:
        response = get_client().head_object(Bucket=BUCKET_NAME, Key=key)
        return response['ContentLength']
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return None
        raise
