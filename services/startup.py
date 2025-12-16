"""Startup health checks for LearnForge."""

import os
import requests
import psycopg2
import boto3
from botocore.config import Config

def check_postgres():
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT", 5432),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            connect_timeout=5
        )
        conn.close()
        return True, None
    except Exception as e:
        return False, str(e).split('\n')[0]

def check_embedding_server():
    url = os.getenv("EMBEDDING_SERVER_URL", "http://localhost:5001")
    try:
        r = requests.get(f"{url}/health", timeout=5)
        return r.status_code == 200, None
    except Exception as e:
        print(f"Embedding server error: {e}")
        return False, "Start with: python embedserver.py"

def check_reranker_server():
    url = os.getenv("RERANK_SERVER_URL", "http://localhost:5002")
    try:
        r = requests.get(f"{url}/health", timeout=5)
        return r.status_code == 200, None
    except Exception as e:
        print(f"Reranker server error: {e}")
        return False, "Start with: python rerankserver.py"

def check_s3_storage():
    try:
        client = boto3.client(
            's3',
            endpoint_url=os.getenv('S3_ENDPOINT_URL'),
            aws_access_key_id=os.getenv('S3_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('S3_SECRET_KEY'),
            region_name=os.getenv('S3_REGION', 'auto'),
            config=Config(signature_version='s3v4', connect_timeout=5)
        )
        client.head_bucket(Bucket=os.getenv('S3_BUCKET_NAME'))
        return True, None
    except Exception as e:
        return False, str(e).split(':')[-1].strip() if ':' in str(e) else str(e)

def run_startup_checks():
    checks = [
        ("PostgreSQL", check_postgres, True),
        ("Embedding Server (5001)", check_embedding_server, False),
        ("Reranker Server (5002)", check_reranker_server, False),
        ("S3 Storage", check_s3_storage, True),
    ]

    failed_critical = False

    for name, check_fn, is_critical in checks:
        ok, err = check_fn()
        if ok:
            print(f"[OK] {name}")
        elif is_critical:
            print(f"[XX] {name} - {err}")
            failed_critical = True
        else:
            print(f"[!!] {name} - {err}")

    return not failed_critical