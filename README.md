# RAG Chat Bot

A simple, **100% local** and **completely free** RAG (Retrieval-Augmented Generation) chatbot. Upload documents and ask questions — the bot answers based on the document content.

## Features

- 📄 Supports PDF, TXT, DOCX, and Markdown files
- 🔍 Semantic search using sentence-transformers embeddings
- 📚 Vector storage with FAISS (exact similarity search)
- 💬 Two interfaces: Web UI (Streamlit) or CLI
- 🤖 Local LLM via Ollama (with HuggingFace Transformers fallback)
- 💾 Save and load vector stores to disk
- 100% local — no API keys, no paid services, no cloud required

## Quick Start

### Prerequisites

1. **Python 3.8+** (install from [python.org](https://www.python.org/downloads/))
2. **Ollama** (recommended LLM backend)

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install Ollama (Recommended)

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:** Download from [ollama.com/download](https://ollama.com/download)

Start Ollama and pull a model:
```bash
ollama serve            # Run in a separate terminal
ollama pull phi3      # Download model (~4GB)
```

> **Alternative:** Use the `google/flan-t5-base` model via the transformers backend
> (no Ollama needed). Add `transformers` and `torch` to your pip install.

### Step 3: Run the Chat Bot

**Web Interface (recommended):**
```bash
streamlit run app.py
```

**Command-Line:**
```bash
python cli.py
```

Or use the setup script for automated setup:
```bash
./setup.sh
```

## Usage

### Web Interface

1. Open `http://localhost:8501` in your browser
2. Configure LLM settings in the sidebar
3. Click **Reinitialize Bot**
4. Upload documents using the file uploader
5. Ask questions in the chat box

### CLI Interface

```bash
python cli.py
```

Commands:
| Command | Description |
|---------|-------------|
| `/upload <path>` | Upload a document |
| `/text <content>` | Add raw text |
| `/query <question>` | Ask a question |
| `/clear` | Clear all documents |
| `/stats` | Show bot statistics |
| `/save <path>` | Save vector store |
| `/load <path>` | Load vector store |
| `/backend <name>` | Switch LLM backend |
| `/model <name>` | Switch LLM model |
| `/help` | Show all commands |
| `/quit` | Exit |

Just type a question directly (no `/` prefix) to chat.

## How It Works

1. **Document Processing**: Documents are split into overlapping chunks (default: 512 tokens, 50-token overlap)
2. **Embedding**: Each chunk is converted to a 384-dimension vector using `all-MiniLM-L6-v2`
3. **Vector Storage**: Vectors are stored in a FAISS `IndexFlatL2` index for exact similarity search
4. **Retrieval**: On query, the 5 most similar chunks are retrieved (configurable)
5. **Generation**: The LLM generates an answer grounded in the retrieved context

## Configuration

### LLM Backends

| Backend | Example Models | Quality |
|---------|---------------|---------|
| **Ollama** (recommended) | `llama3`, `gemma2`, `phi3`, `mistral`, `qwen2` | Excellent |
| **Transformers** (fallback) | `google/flan-t5-base`, `google/flan-t5-large` | Good |

### CLI Options

```bash
python cli.py --backend ollama --model llama3 --top-k 5 --temperature 0.3
python cli.py --backend transformers --model google/flan-t5-base
```

### Environment Variables

No environment variables or API keys are required. All models run locally.

## File Structure

```
rag-chtbot/
├── app.py           # Streamlit web application
├── cli.py           # Command-line interface
├── rag_bot.py       # Core RAG engine
├── requirements.txt # Python dependencies
├── setup.sh         # Automated setup script
└── README.md        # This file
```

## Tips & Tricks

- The first run will download the embedding model (~80 MB) from HuggingFace
- Make sure Ollama is running before starting: `ollama serve`
- For better quality: use `llama3` or `gemma2` with Ollama
- For faster startup: use a smaller Ollama model like `phi3` (~800 MB)
- Use `--top-k` to control how many chunks are retrieved (higher = more context, slower)
- Adjust `--temperature` (0.0 = deterministic, 1.0 = creative)
- The save/load feature persists your vector store between sessions

## Troubleshooting

**Ollama connection errors**
```
Warning: Could not reach Ollama
```
Solution: Start the Ollama server: `ollama serve` (keep it running in a separate terminal)

**Model not found**
```
Error: model 'xxx' not found
```
Solution: Pull the model first: `ollama pull phi3`

**Unsupported file type**
Make sure the file extension is `.pdf`, `.txt`, `.docx`, or `.md`

**Slow embeddings on first run**
The sentence-transformers model (~80 MB) downloads on first use. Subsequent runs are fast.

**Empty responses**
Check that the document was uploaded successfully (`/stats` in CLI or check the sidebar in the web app).

## License

MIT License — use freely for personal and commercial projects.
