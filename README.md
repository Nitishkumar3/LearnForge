# LearnForge

A smart campus assistant for students to upload course materials, ask questions with cited answers, and leverage multiple AI models for learning.

## Features

### Core Features
- **Multi-User Authentication** - Sign up, sign in, email verification, password reset
- **Workspaces** - Organize documents and conversations into separate workspaces
- **PDF Upload** - Drag & drop PDFs with automatic processing
- **Q&A with Citations** - Ask questions, get answers with source references [1], [2]
- **Conversation History** - ChatGPT-style conversation management
- **Streaming Responses** - Real-time response streaming via SSE

### AI & Search
- **Multi-LLM Support** - Gemini, Groq, Mistral, Cerebras (switch models on-the-fly)
- **Hybrid Search** - BM25 keyword + vector semantic + reranking
- **Local Embeddings** - BGE-M3 ONNX model (no API costs)
- **Web Search** - Real-time web search integration (per-query toggle)
- **Extended Thinking** - Reasoning mode for complex questions
- **Image Generation** - Amazon Nova Canvas via AWS Bedrock

### Frontend
- **Dark/Light Mode** - Smooth theme switching with persistence
- **Markdown Rendering** - Full CommonMark with code highlighting
- **LaTeX Math** - KaTeX for mathematical equations
- **Text-to-Speech** - Read responses aloud
- **Responsive Design** - Mobile-friendly UI

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Flask (Python) |
| Database | PostgreSQL |
| Vector DB | ChromaDB (persistent) |
| Embeddings | BGE-M3 (ONNX) - Local |
| Storage | AWS S3 / Cloudflare R2 |
| LLM | Gemini, Groq, Mistral, Cerebras |
| Image Gen | Amazon Nova Canvas (Bedrock) |
| Frontend | Tailwind CSS, Google Sans Flex |

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy example env file
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Required API Keys
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
MISTRAL_API_KEY=your_mistral_key
CEREBRAS_API_KEY=your_cerebras_key

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=learnforge
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

# JWT Secret
JWT_SECRET_KEY=your_secret_key

# S3/R2 Storage
S3_ENDPOINT_URL=https://your-account.r2.cloudflarestorage.com
S3_ACCESS_KEY=your_access_key
S3_SECRET_KEY=your_secret_key
S3_BUCKET_NAME=learnforge
S3_REGION=auto

# SMTP (for email verification)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_FROM_EMAIL=your_email@gmail.com
APP_URL=http://localhost:5000
```

### 3. Setup Database

```bash
# Create PostgreSQL database
psql -U postgres -c "CREATE DATABASE learnforge;"

# Run schema
psql -U postgres -d learnforge -f schema.sql
```

### 4. Download BGE-M3 Model

Download the ONNX model and place in `bge-m3-onnx/` folder:
- `model.onnx`
- `model.onnx_data`
- Tokenizer files from HuggingFace

### 5. Run Services

**Terminal 1 - Embedding Server:**
```bash
python embedserver.py
# Runs on port 5001
```

**Terminal 2 - Rerank Server (optional):**
```bash
python rerankserver.py
# Runs on port 5002
```

**Terminal 3 - Main App:**
```bash
python app.py
# Runs on port 5000
```

### 6. Open Browser

Navigate to `http://localhost:5000`

## Project Structure

```
LearnForge/
├── app.py                 # Main Flask app (routes, API endpoints)
├── embedserver.py         # Local embedding server (BGE-M3 ONNX)
├── rerankserver.py        # Reranking server for hybrid search
├── schema.sql             # PostgreSQL database schema
├── requirements.txt       # Python dependencies
├── .env.example           # Configuration template
├── design-guide.md        # UI/UX design system documentation
│
├── services/
│   ├── auth.py            # JWT authentication, password hashing
│   ├── database.py        # PostgreSQL connection and queries
│   ├── embedding.py       # Embedding service client
│   ├── generation.py      # LLM answer generation with context
│   ├── email.py           # SMTP email (verification, password reset)
│   ├── llm_providers.py   # Multi-LLM support (Gemini, Groq, etc.)
│   ├── retrieval.py       # Hybrid search (BM25 + Vector + Rerank)
│   ├── storage.py         # S3/R2 file operations
│   ├── startup.py         # Health checks on startup
│   └── vector_store.py    # ChromaDB operations
│
├── processors/
│   └── pdf_processor.py   # PDF text extraction
│
├── utils/
│   └── chunking.py        # Smart text chunking
│
├── templates/
│   ├── dashboard.html     # Main app UI (ChatGPT-style)
│   ├── index.html         # Landing page
│   ├── signin.html        # Sign in page
│   ├── signup.html        # Sign up page
│   ├── forgotpassword.html
│   └── resetpassword.html
│
├── bge-m3-onnx/           # ONNX model files
│   └── model.onnx
│
└── data/                  # Auto-created
    ├── chroma_db/         # Vector database (persistent)
    └── uploads/           # Temporary PDF uploads
```

## API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/signup` | Register new user |
| POST | `/api/auth/signin` | Login (returns JWT) |
| GET | `/api/auth/verify-email/<token>` | Verify email |
| POST | `/api/auth/forgot-password` | Request password reset |
| POST | `/api/auth/reset-password` | Reset password |
| GET | `/api/auth/me` | Get current user |

### Workspaces
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/workspaces` | List workspaces |
| POST | `/api/workspaces` | Create workspace |
| GET | `/api/workspaces/<id>` | Get workspace |
| PUT | `/api/workspaces/<id>` | Update workspace |
| DELETE | `/api/workspaces/<id>` | Delete workspace |

### Conversations
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/conversations` | List conversations |
| POST | `/api/conversations` | Create conversation |
| GET | `/api/conversations/<id>` | Get with messages |
| PUT | `/api/conversations/<id>` | Update title |
| DELETE | `/api/conversations/<id>` | Delete conversation |
| POST | `/api/conversations/<id>/chat` | Send message (SSE) |
| POST | `/api/conversations/<id>/title` | Auto-generate title |
| DELETE | `/api/workspaces/<id>/conversations` | Bulk delete |

### Documents
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/documents` | List documents |
| POST | `/api/upload` | Upload PDF |
| DELETE | `/api/documents/<id>` | Delete document |
| DELETE | `/api/workspaces/<id>/documents` | Bulk delete |

### Images
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/generate-image` | Generate image (Nova Canvas) |
| GET | `/api/images/<s3_key>` | Serve image (proxy) |
| GET | `/api/images/<s3_key>/download` | Download image |

### Utility
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stats` | User statistics |
| GET | `/api/models` | List LLM models |
| POST | `/api/models/switch` | Switch active model |

## Database Schema

### Tables
- **users** - User accounts with email verification
- **workspaces** - Multi-tenant workspace isolation
- **documents** - PDF metadata (pages, chunks, S3 path)
- **conversations** - Chat sessions per workspace
- **messages** - User/assistant messages with sources

### Key Features
- UUID primary keys
- Automatic timestamps (created_at, updated_at)
- Cascade deletes for data integrity
- JSONB for sources and attachments
- Indexes on frequently queried columns

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | - | Google Gemini API key |
| `GROQ_API_KEY` | - | Groq API key |
| `MISTRAL_API_KEY` | - | Mistral API key |
| `CEREBRAS_API_KEY` | - | Cerebras API key |
| `EMBEDDING_SERVER_URL` | localhost:5001 | Embedding server URL |
| `RERANK_SERVER_URL` | localhost:5002 | Rerank server URL |
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `MAX_CONTEXT_TOKENS` | 7000 | Max tokens for context |
| `MAX_CONTENT_LENGTH` | 100MB | Max upload size |

## How It Works

### Document Processing
1. **Upload** - PDF uploaded to server
2. **Extract** - Text extracted from all pages
3. **Chunk** - Smart chunking with sentence awareness
4. **Embed** - Local BGE-M3 generates 1024-dim vectors
5. **Store** - Vectors in ChromaDB, PDF in S3

### Query Flow
1. **Embed** - Question converted to vector
2. **Search** - Hybrid BM25 + vector + reranking
3. **Context** - Top chunks selected (token-limited)
4. **Generate** - LLM generates answer with citations
5. **Stream** - Response streamed via SSE
6. **Save** - Message saved to database

### Theme System
- Theme stored in `localStorage` as `learnforge_theme`
- CSS class on `<html>` element (`light` or `dark`)
- Tailwind dark mode variants (`dark:bg-dark-bg`)
- Smooth 0.2s transitions on all elements

## Available LLM Models

| Model | Provider | Features |
|-------|----------|----------|
| Gemini 2.5 Flash | Google | Web search, extended thinking |
| Gemini 2.5 Flash Lite | Google | Fast, reasoning |
| Mistral Medium | Mistral | Web search |
| Groq Compound | Groq | Web search |
| Llama 4 Scout 17B | Groq | Fast inference |
| Llama 3.3 70B | Cerebras | Ultra-fast |
| Nova Canvas | AWS Bedrock | Image generation |

## Requirements

- Python 3.10+
- PostgreSQL 14+
- ~2GB disk space for BGE-M3 model
- Node.js (optional, for Tailwind development)

## License

MIT License
