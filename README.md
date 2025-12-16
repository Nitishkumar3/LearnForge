# LearnForge

An AI-powered intelligent learning platform that functions as a smart campus assistant. Upload course materials (PDFs, documents, images, audio/video), ask context-aware questions with cited answers, and leverage multiple AI models for enhanced learning through RAG (Retrieval-Augmented Generation).

## Table of Contents

- [Screenshots](SCREENSHOTS.md)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Application Setup](#application-setup)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Database Schema](#database-schema)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Available LLM Models](#available-llm-models)
- [Supported File Formats](#supported-file-formats)

## Features

### Core Learning Features
- **Multi-Document RAG System** - Upload and process multiple documents with intelligent retrieval
- **Q&A with Citations** - Ask questions, get answers with source references [1], [2], etc.
- **Conversation History** - ChatGPT-style conversation management with auto-generated titles
- **Streaming Responses** - Real-time response streaming via Server-Sent Events (SSE)

### Study Tools
- **Study Materials Generation** - AI-generated comprehensive study guides from document content
- **Flash Cards** - Auto-generated flashcards for spaced repetition learning
- **Interactive Quizzes** - Multiple quiz types (MCQ, Fill-in-the-Blank, Subjective) with AI-powered question generation
- **Mind Maps** - Visual mind map generation for concept understanding
- **Syllabus Management** - Upload and organize course syllabi

### Document Processing
- **Multi-Format Support** - PDF, DOCX, PPTX, XLSX, Images (PNG, JPG), Audio (MP3, WAV), Video (MP4, MOV)
- **Smart OCR** - Automatic OCR detection for image-heavy PDFs with Gemini Vision
- **Audio/Video Transcription** - Whisper-powered speech-to-text

### AI & Search
- **Multi-LLM Support** - Switch between models on-the-fly:
  - Google Gemini 2.5 Flash (with web search & extended thinking)
  - Gemini 2.5 Flash Lite
  - Mistral Medium (with web search)
  - Groq Compound (with web search)
  - Llama models (via Groq & Cerebras)
  - ZAI GLM 4.6
- **Hybrid Search** - BM25 keyword + vector semantic + reranking
- **Local Embeddings** - BGE-M3 ONNX model (1024-dim vectors, no API costs)
- **Web Search** - Exa.ai Deep Search integration for real-time information
- **Deep Research** - Comprehensive research mode powered by Exa.ai for in-depth topic exploration
- **Query Rewriting** - Automatic query expansion for better retrieval
- **Image Generation** - Amazon Nova Canvas via AWS Bedrock

### User & Workspace Management
- **Multi-User Authentication** - Sign up, email verification, password reset
- **Workspaces** - Multi-tenant workspace isolation for organizing content
- **Document Tagging** - Organize documents with custom tags

### Frontend
- **Dark/Light Mode** - Smooth theme switching with persistence
- **Markdown Rendering** - Full CommonMark with code syntax highlighting
- **LaTeX Math** - KaTeX for mathematical equations
- **Mermaid Diagrams** - Flowcharts and diagrams support
- **Text-to-Speech** - Read responses aloud

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Flask (Python 3.10+) |
| Database | PostgreSQL 14+ |
| Vector DB | ChromaDB (persistent) |
| Embeddings | BGE-M3 ONNX (local inference) |
| Reranking | MS-MARCO MiniLM |
| Storage | AWS S3 / Cloudflare R2 |
| LLM Providers | Google Gemini, Groq, Mistral, Cerebras |
| Image Generation | Amazon Nova Canvas (Bedrock) |
| Web Search | Exa.ai |
| Frontend | Tailwind CSS, Google Sans Flex |
| Auth | JWT + bcrypt |

## Application Setup

### Prerequisites
- Python 3.10+
- PostgreSQL 14+
- ~2GB disk space for BGE-M3 model
- GPU with minimum 4GB VRAM for Embedding and Re Ranker 
- API keys for all the providers (Groq, Google AI Studio, Cerebras and Mistral).

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

### 2. Configure Environment Variables

```bash
# Copy example env file
cp .env.example .env
```

Edit `.env` with your configuration.

also set up `cookie.txt` file with exported AWS Party Rock Cookies. 
 
### 3. Setup Database

```bash
# Create PostgreSQL database
psql -U postgres -c "CREATE DATABASE learnforge;"

# Run schema
psql -U postgres -d learnforge -f schema.sql
```

### 4. Download ONNX Models

**BGE-M3 Embedding Model:**

Download from HuggingFace ([BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)) and place in `bge-m3-onnx/`:
- `model.onnx` (~2GB)
- `model.onnx_data`
- Tokenizer files (config.json, tokenizer.json, special_tokens_map.json, tokenizer_config.json)

**MS-MARCO MiniLM Reranker Model:**

Download from HuggingFace ([cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)) and place in `ms-marco-MiniLM-L-6-v2/`:
- `model.onnx`
- Tokenizer files (config.json, tokenizer.json, special_tokens_map.json, tokenizer_config.json, vocab.txt)

### 5. Run Services

**Terminal 1 - Embedding Server (Port 5001):**
```bash
python embedserver.py
```

**Terminal 2 - Rerank Server (Port 5002):**
```bash
python rerankserver.py
```

**Terminal 3 - Main App (Port 5000):**
```bash
python app.py
```

### 6. Open Browser

Navigate to `http://localhost:5000`

## Project Structure

```
LearnForge/
├── app.py                    # Main Flask app (routes, API endpoints)
├── embedserver.py            # Local embedding server (BGE-M3 ONNX)
├── rerankserver.py           # Reranking server (MS-MARCO)
├── schema.sql                # PostgreSQL database schema
├── requirements.txt          # Python dependencies
├── .env.example              # Configuration template
│
├── services/
│   ├── auth.py               # JWT authentication, password hashing
│   ├── database.py           # PostgreSQL connection and queries
│   ├── embedding.py          # Embedding service client
│   ├── generation.py         # LLM answer generation with context
│   ├── email.py              # SMTP email (verification, password reset)
│   ├── llm_providers.py      # Multi-LLM support (Gemini, Groq, etc.)
│   ├── retrieval.py          # Hybrid search (BM25 + Vector + Rerank)
│   ├── storage.py            # S3/R2 file operations
│   ├── startup.py            # Health checks on startup
│   ├── vector_store.py       # ChromaDB operations
│   └── exa_search.py         # Exa.ai web search integration
│
├── processors/
│   ├── file_types.py         # File type registry & validation
│   ├── pdf.py                # PDF extraction with auto OCR detection
│   ├── word.py               # DOCX processing
│   ├── presentation.py       # PPTX processing
│   ├── excel.py              # XLSX/CSV processing
│   ├── image.py              # Image processing
│   ├── audio.py              # Audio/video extraction
│   ├── whisper.py            # Speech-to-text transcription
│   └── gemini_ocr.py         # Gemini Vision OCR
│
├── utils/
│   └── chunking.py           # Smart text chunking with sentence awareness
│
├── templates/
│   ├── index.html            # Landing page
│   ├── chat.html             # Main chat interface
│   ├── study.html            # Study materials viewer
│   ├── flashcards.html       # Flashcard UI
│   ├── quiz.html             # Quiz interface
│   ├── uploads.html          # Document management
│   ├── view.html             # Document viewer
│   ├── signin.html           # Sign in page
│   ├── signup.html           # Sign up page
│   ├── forgotpassword.html   # Password reset request
│   └── resetpassword.html    # Password reset form
│
├── static/
│   └── js/navbar.js          # Navigation JavaScript
│
├── bge-m3-onnx/              # ONNX model files (~2GB)
│   ├── model.onnx
│   ├── model.onnx_data
│   └── [tokenizer files]
│
└── data/                     # Auto-created runtime data
    ├── chroma_db/            # Vector database (persistent)
    └── uploads/              # Temporary uploads
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
| POST | `/api/conversations/<id>/chat` | Send message (SSE streaming) |
| POST | `/api/conversations/<id>/title` | Auto-generate title |
| POST | `/api/workspaces/<id>/conversations/delete` | Bulk delete |

### Documents
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/documents` | List documents |
| POST | `/api/upload` | Upload document (multipart) |
| DELETE | `/api/documents/<id>` | Delete document |
| POST | `/api/documents/<id>/analyze` | Analyze OCR recommendation |
| GET | `/api/documents/<id>/summary` | Get AI summary |
| GET | `/api/documents/<id>/content` | Get extracted text |

### Study Materials
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/workspaces/<id>/study` | List study materials |
| POST | `/api/workspaces/<id>/study/generate` | Generate study guide |
| POST | `/api/workspaces/<id>/study/generate-stream` | Streaming generation |
| GET | `/api/workspaces/<id>/study/<module>/<subtopic>` | Get content |

### Flash Cards
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/workspaces/<id>/flashcards` | List flashcard sets |
| POST | `/api/workspaces/<id>/flashcards/generate` | Generate flashcards |
| GET | `/api/workspaces/<id>/flashcards/<module>/<subtopic>` | Get cards |

### Quizzes
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/workspaces/<id>/quiz/generate` | Generate quiz (mcq/fitb/subjective) |
| GET | `/api/quizzes/<id>` | Get quiz |
| POST | `/api/quizzes/<id>/submit` | Submit answers |

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
| GET | `/api/models` | List available LLM models |
| POST | `/api/models/switch` | Switch active model |
| GET | `/api/upload-analysis` | Check supported formats |

## Database Schema

### Tables
- **users** - User accounts with email verification
- **workspaces** - Multi-tenant workspace isolation with syllabus support
- **documents** - Document metadata (pages, chunks, S3 path, processing status)
- **conversations** - Chat sessions per workspace
- **messages** - User/assistant messages with sources and thinking
- **study_materials** - AI-generated study content by module/subtopic
- **flash_cards** - Flashcard sets with question/answer/difficulty
- **quizzes** - Quiz data with questions, answers, and results

### Key Features
- UUID primary keys
- Automatic timestamps (created_at, updated_at)
- Cascade deletes for data integrity
- JSONB for sources, cards, questions, and attachments
- Indexes on frequently queried columns

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | - | Google Gemini API key |
| `GROQ_API_KEY` | - | Groq API key |
| `MISTRAL_API_KEY` | - | Mistral API key |
| `CEREBRAS_API_KEY` | - | Cerebras API key |
| `EXA_API_KEY` | - | Exa.ai web search API key |
| `EMBEDDING_SERVER_URL` | localhost:5001 | Embedding server URL |
| `RERANK_SERVER_URL` | localhost:5002 | Rerank server URL |
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `MAX_CONTEXT_TOKENS` | 7000 | Max tokens for LLM context |
| `MAX_CONTENT_LENGTH` | 100MB | Max upload size |
| `RERANK_SCORE_THRESHOLD` | 0 | Minimum reranking score |

## How It Works

### Document Processing Pipeline
1. **Upload** - Document uploaded to server
2. **Validate** - File type detection and validation
3. **Extract** - Text extracted using appropriate processor (PDF, DOCX, etc.)
4. **OCR** (if needed) - Gemini Vision for image-heavy documents
5. **Chunk** - Smart chunking with sentence awareness and overlap
6. **Embed** - Local BGE-M3 generates 1024-dim vectors
7. **Store** - Vectors in ChromaDB, files in S3/R2

### Query Flow
1. **Rewrite** - Query expanded with synonyms via LLM
2. **Embed** - Question converted to vector
3. **Search** - Hybrid BM25 + vector search
4. **Fuse** - Reciprocal rank fusion of results
5. **Rerank** - MS-MARCO reranker scores relevance
6. **Context** - Top chunks selected (token-limited)
7. **Generate** - LLM generates answer with citations
8. **Stream** - Response streamed via SSE
9. **Save** - Message saved to database

### Study Material Generation
1. **Analyze** - Parse syllabus structure
2. **Retrieve** - Get relevant content for each module/subtopic
3. **Generate** - LLM creates comprehensive study guide
4. **Store** - Save to database for future access

## Available LLM Models

| Model | Provider | Features |
|-------|----------|----------|
| Gemini 2.5 Flash | Google | Web search, extended thinking |
| Gemini 2.5 Flash Lite | Google | Fast, cost-effective |
| Mistral Medium | Mistral | Web search |
| Groq Compound | Groq | Web search |
| Llama 4 Scout 17B | Groq | Fast inference |
| Llama 3.3 70B | Cerebras | Ultra-fast inference |
| ZAI GLM 4.6 | ZAI | Multilingual support |
| Nova Canvas | AWS Bedrock | Image generation |

## Supported File Formats

| Category | Formats |
|----------|---------|
| Documents | PDF, DOCX, PPTX, XLSX, CSV |
| Images | PNG, JPG, JPEG, GIF, WebP |
| Audio | MP3, WAV, M4A, OGG, FLAC |
| Video | MP4, MOV, AVI, MKV, WebM |