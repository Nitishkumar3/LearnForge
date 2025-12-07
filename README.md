# LearnForge

A smart document assistant for students to upload course materials and ask questions with cited answers.

## Features

- **PDF Upload** - Drag & drop PDFs to upload
- **Q&A with Citations** - Ask questions, get answers with source references [1], [2]
- **Local Embeddings** - BGE-M3 model runs locally (no API costs for embeddings)
- **Persistent Storage** - ChromaDB stores vectors across restarts
- **Chat History** - Maintains conversation context

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Flask |
| Embeddings | BGE-M3 (ONNX) - Local |
| Generation | Gemini API |
| Vector DB | ChromaDB |
| Frontend | Tailwind CSS |

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your Gemini API key
GEMINI_API_KEY=your_key_here
```

Get Gemini API key: https://aistudio.google.com/app/apikey

### 3. Download BGE-M3 Model

Download the ONNX model and place in `bge-m3-onnx/` folder:
- `model.onnx`
- `tokenizer.json` (or use HuggingFace tokenizer)

### 4. Run

**Terminal 1 - Embedding Server:**
```bash
python embedserver.py
```

**Terminal 2 - Main App:**
```bash
python app.py
```

### 5. Open Browser

Navigate to `http://localhost:5000`

## Project Structure

```
LearnForge/
├── app.py                 # Main Flask app
├── embedserver.py         # Local embedding server (BGE-M3)
├── requirements.txt       # Dependencies
├── .env                   # Configuration (create from .env.example)
│
├── services/
│   ├── embedding.py       # Embedding service client
│   ├── generation.py      # Gemini generation
│   ├── retrieval.py       # Vector retrieval
│   └── vector_store.py    # ChromaDB operations
│
├── processors/
│   └── pdf_processor.py   # PDF text extraction
│
├── utils/
│   └── chunking.py        # Text chunking
│
├── templates/
│   └── index.html         # Frontend UI
│
├── bge-m3-onnx/           # ONNX model files
│   └── model.onnx
│
└── data/                  # Auto-created
    ├── chroma_db/         # Vector database
    ├── uploads/           # Uploaded PDFs
    └── documents.json     # Document registry
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main UI |
| GET | `/api/documents` | List uploaded documents |
| POST | `/api/upload` | Upload PDF |
| DELETE | `/api/documents/:id` | Delete document |
| POST | `/api/chat` | Send question |
| POST | `/api/clear` | Clear all data |
| GET | `/api/stats` | System statistics |

## Configuration

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | - | Required for generation |
| `EMBEDDING_SERVER_URL` | localhost:5001 | Embedding server |
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `TOP_K` | 15 | Chunks to retrieve |

## How It Works

1. **Upload** - PDF text extracted and split into chunks
2. **Embed** - Each chunk converted to 1024-dim vector (BGE-M3)
3. **Store** - Vectors stored in ChromaDB with metadata
4. **Query** - Question embedded, similar chunks retrieved
5. **Generate** - Gemini generates answer using retrieved context
6. **Cite** - Sources shown with page numbers

## Requirements

- Python 3.10+
- CUDA (optional, for GPU acceleration)
- ~2GB disk space for BGE-M3 model
