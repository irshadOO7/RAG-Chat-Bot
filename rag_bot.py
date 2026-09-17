"""
rag_bot.py - Local RAG (Retrieval-Augmented Generation) Chatbot Engine

A simple, fully-local RAG implementation using:
  - sentence-transformers for text embeddings
  - FAISS for vector similarity search
  - Ollama for LLM inference (with transformers fallback)

Free and 100% local - no API keys or paid services required.

Usage:
    bot = RAGBot()                        # Ollama with llama3 by default
    bot.add_document("doc.pdf")           # Upload a document
    result = bot.query("What is it about?")  # Ask a question
    print(result.answer)                  # Get the answer
"""


import os

# Set thread limits BEFORE importing torch/sentence-transformers to avoid
# segfaults from OpenMP/MKL thread conflicts on some platforms.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import json
import faiss
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import pypdf
import docx


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A text chunk extracted from a document, stored in the vector index."""
    text: str
    source: str
    chunk_index: int


@dataclass
class QueryResult:
    """Result of a RAG query."""
    answer: str
    sources: List[str]
    context: str
    retrieved_chunks: List[Tuple[Chunk, float]]



# ---------------------------------------------------------------------------
# Core RAG engine
# ---------------------------------------------------------------------------

class RAGBot:
    """
    A local RAG chatbot that answers questions based on uploaded documents.

    The pipeline is:
        1. Split documents into overlapping chunks
        2. Embed each chunk with sentence-transformers
        3. Store vectors in a FAISS index
        4. On query: retrieve top-K similar chunks, then generate an answer
    """

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_LLM_MODEL = "llama3"
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_TOP_K = 5
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    PROMPT_TEMPLATE = (
        "You are a helpful AI assistant. Answer the question using only the "
        "context provided below. If you cannot answer from the context, say "
        "\"I don't have enough information to answer that question.\""
        "\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        llm_backend: str = "ollama",
        llm_model: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.embedding_model_name = embedding_model or self.EMBEDDING_MODEL
        self.llm_backend = llm_backend
        self.llm_model = llm_model or self.DEFAULT_LLM_MODEL
        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or self.DEFAULT_CHUNK_OVERLAP
        self.temperature = temperature
        self.top_k = top_k

        print(f"[RAGBot] Loading embedding model: {self.embedding_model_name}")
        self.embedder = SentenceTransformer(self.embedding_model_name)
        self.embed_dim = self.embedder.get_sentence_embedding_dimension()
        print(f"[RAGBot] Embedding dimension: {self.embed_dim}")

        self.index = faiss.IndexFlatL2(self.embed_dim)
        self.chunks: List[Chunk] = []
        self._chunk_sources: set = set()

        self._ollama = None
        self._generator = None
        if llm_backend is not None:
            self._init_llm()


    # ------------------------------------------------------------------
    # LLM initialization
    # ------------------------------------------------------------------

    def _init_llm(self):
        """Initialize or re-initialize the LLM backend."""
        if self.llm_backend == "ollama":
            import ollama
            self._ollama = ollama
            self._generator = None
            try:
                self._ollama.chat(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": "Hello"}],
                    options={"temperature": self.temperature},
                )
                print(f"[RAGBot] Connected to Ollama (model: {self.llm_model})")
            except Exception as e:
                print(f"[RAGBot] Warning: Could not reach Ollama: {e}")
                print(f"[RAGBot] Run `ollama serve` and "
                      f"`ollama pull {self.llm_model}`")
                self._ollama = None

        elif self.llm_backend == "transformers":
            from transformers import (
                pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
            )
            print(f"[RAGBot] Loading transformers model: {self.llm_model}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_model)
            self._generator = pipeline(
                "text2text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
            )
            self._ollama = None
            print(f"[RAGBot] Transformers model ready: {self.llm_model}")
        else:
            raise ValueError(f"Unknown LLM backend: {self.llm_backend}")

    # ------------------------------------------------------------------
    # Text chunking
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str, source: str) -> List[Chunk]:
        """
        Split *text* into overlapping chunks.

        Approximates 1 token = 4 characters (reasonable for English).
        """
        chunk_chars = self.chunk_size * 4
        overlap_chars = self.chunk_overlap * 4
        chunks: List[Chunk] = []

        if len(text) <= chunk_chars:
            chunks.append(Chunk(text=text.strip(), source=source, chunk_index=0))
            return chunks

        step = max(1, chunk_chars - overlap_chars)
        idx = 0
        for start in range(0, len(text), step):
            end = min(start + chunk_chars, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(text=chunk_text, source=source, chunk_index=idx))
                idx += 1

        return chunks


    # ------------------------------------------------------------------
    # Document / text loading
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pdf(filepath: str) -> str:
        """Extract text from a PDF file using pypdf."""
        reader = pypdf.PdfReader(filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    @staticmethod
    def _extract_docx(filepath: str) -> str:
        """Extract text from a DOCX file using python-docx."""
        doc = docx.Document(filepath)
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(paragraphs)

    @staticmethod
    def _extract_txt(filepath: str) -> str:
        """Extract text from a plain-text file."""
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    @classmethod
    def _detect_file_type(cls, filepath: str) -> str:
        """Detect the file type from its extension."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            return "pdf"
        if ext in (".docx", ".doc"):
            return "docx"
        if ext in (".txt", ".md"):
            return "txt"
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported: {', '.join(cls.SUPPORTED_EXTENSIONS)}"
        )

    def add_document(self, filepath: str, source_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a document from disk, chunk it, embed it, and add to the index.

        Args:
            filepath: Path to the document.
            source_name: Custom name for the document (defaults to filename).

        Returns:
            Metadata dict about the processed document.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        source_name = source_name or os.path.basename(filepath)
        file_type = self._detect_file_type(filepath)

        if file_type == "pdf":
            text = self._extract_pdf(filepath)
        elif file_type == "docx":
            text = self._extract_docx(filepath)
        else:
            text = self._extract_txt(filepath)

        if not text.strip():
            print(f"[RAGBot] Warning: '{source_name}' appears to be empty.")

        return self.add_text(text, source_name)

    def add_text(self, text: str, source_name: str = "text") -> Dict[str, Any]:
        """
        Add raw text content directly (no file needed).

        Args:
            text: The text content.
            source_name: Identifier for this text source.

        Returns:
            Metadata dict about the processed chunks.
        """
        chunks = self._chunk_text(text, source_name)

        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(
            chunk_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        self.index.add(embeddings)

        start_idx = len(self.chunks)
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = start_idx + i
            self.chunks.append(chunk)

        self._chunk_sources.add(source_name)

        return {
            "source": source_name,
            "num_chunks": len(chunks),
            "num_chars": len(text),
            "total_chunks": len(self.chunks),
        }


    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Chunk, float]]:
        """
        Retrieve the most relevant chunks for *query*.

        Returns a list of (Chunk, similarity_score) tuples where the
        similarity score is in [0, 1] (1 = most similar).
        """
        top_k = top_k or self.top_k

        query_embedding = self.embedder.encode(
            [query], show_progress_bar=False, convert_to_numpy=True
        )

        distances, indices = self.index.search(query_embedding, top_k)

        results: List[Tuple[Chunk, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            similarity = float(1.0 / (1.0 + dist))
            results.append((chunk, similarity))

        return results

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_ollama(self, prompt: str) -> str:
        """Generate a response using the Ollama backend."""
        if self._ollama is None:
            self._init_llm()
            if self._ollama is None:
                raise RuntimeError(
                    "Ollama is not available. Make sure the Ollama server is "
                    f"running (`ollama serve`) and the model '{self.llm_model}' "
                    f"is pulled (`ollama pull {self.llm_model}`)."
                )

        response = self._ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature, "top_p": 0.9},
        )

        if hasattr(response, "message"):
            return response.message.content.strip()
        return response["message"]["content"].strip()

    def _generate_transformers(self, prompt: str) -> str:
        """Generate a response using the HuggingFace Transformers backend."""
        if self._generator is None:
            self._init_llm()

        result = self._generator(
            prompt,
            max_length=1024,
            temperature=self.temperature,
            do_sample=True,
        )
        return result[0]["generated_text"].strip()

    def generate(self, query: str, context: str) -> str:
        """
        Generate a response from *query* and *context*.

        Args:
            query: The user's question.
            context: Retrieved context chunks joined as a string.

        Returns:
            The generated answer.
        """
        prompt = self.PROMPT_TEMPLATE.format(context=context, question=query)

        if self.llm_backend == "ollama":
            return self._generate_ollama(prompt)
        else:
            return self._generate_transformers(prompt)


    # ------------------------------------------------------------------
    # Full RAG pipeline
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: Optional[int] = None) -> QueryResult:
        """
        Run the full RAG pipeline: retrieve relevant chunks, then generate.

        Args:
            question: The user's question.
            top_k: Override the default number of chunks to retrieve.

        Returns:
            QueryResult containing the answer, sources, context, and
            retrieved chunks.
        """
        top_k = top_k or self.top_k
        retrieved = self.retrieve(question, top_k=top_k)

        if not retrieved:
            return QueryResult(
                answer="I don't have enough information to answer that question.",
                sources=[],
                context="",
                retrieved_chunks=[],
            )

        context_parts: List[str] = []
        sources: List[str] = []
        for chunk, score in retrieved:
            context_parts.append(f"[Score: {score:.3f}]\n{chunk.text}")
            sources.append(chunk.source)

        context = "\n\n---\n\n".join(context_parts)
        answer = self.generate(question, context)

        # Deduplicate sources while preserving order
        seen = set()
        ordered_sources: List[str] = []
        for s in sources:
            if s not in seen:
                seen.add(s)
                ordered_sources.append(s)

        return QueryResult(
            answer=answer,
            sources=ordered_sources,
            context=context,
            retrieved_chunks=retrieved,
        )

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def clear(self):
        """Remove all documents from the vector store."""
        self.index.reset()
        self.chunks.clear()
        self._chunk_sources.clear()
        print("[RAGBot] Vector store cleared.")

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the current state of the bot."""
        return {
            "num_chunks": len(self.chunks),
            "num_sources": len(self._chunk_sources),
            "sources": sorted(self._chunk_sources),
            "embedding_model": self.embedding_model_name,
            "embed_dim": self.embed_dim,
            "llm_backend": self.llm_backend,
            "llm_model": self.llm_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "temperature": self.temperature,
        }

    def save(self, path: str):
        """
        Persist the FAISS index and chunk metadata to *path*.

        Creates:
          - faiss_index.bin : the vector index
          - metadata.json   : chunks + configuration
        """
        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, os.path.join(path, "faiss_index.bin"))

        metadata = {
            "chunks": [asdict(c) for c in self.chunks],
            "config": {
                "embedding_model": self.embedding_model_name,
                "llm_backend": self.llm_backend,
                "llm_model": self.llm_model,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "top_k": self.top_k,
                "temperature": self.temperature,
                "embed_dim": self.embed_dim,
            },
            "stats": self.get_stats(),
        }

        with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"[RAGBot] Saved vector store to '{path}' ({len(self.chunks)} chunks).")

    def load(self, path: str):
        """Load a previously saved FAISS index and chunk metadata from *path*."""
        index_path = os.path.join(path, "faiss_index.bin")
        metadata_path = os.path.join(path, "metadata.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        self.index = faiss.read_index(index_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.chunks = [Chunk(**c) for c in metadata["chunks"]]
        self._chunk_sources = set(c["source"] for c in self.chunks)

        print(f"[RAGBot] Loaded vector store from '{path}' "
              f"({len(self.chunks)} chunks, {len(self._chunk_sources)} sources).")

    def switch_llm(self, backend: str, model: str):
        """Switch the LLM backend and model at runtime."""
        self.llm_backend = backend
        self.llm_model = model
        self._init_llm()



# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=== RAGBot Smoke Test ===")

    try:
        bot = RAGBot()
    except Exception:
        print("[RAGBot] Ollama not available, trying transformers fallback...")
        bot = RAGBot(llm_backend="transformers", llm_model="google/flan-t5-base")

    sample = (
        "The RAG (Retrieval-Augmented Generation) chatbot combines retrieval "
        "and generation. It first retrieves relevant context from a knowledge "
        "base, then uses a language model to generate a response based on "
        "that context. This approach reduces hallucination compared to "
        "pure LLMs, because answers are grounded in retrieved documents."
    )
    bot.add_text(sample, "sample")

    result = bot.query("How does RAG reduce hallucination?")
    print(f"\nQuestion: How does RAG reduce hallucination?")
    print(f"Answer: {result.answer}")
    print(f"Sources: {result.sources}")

    print(f"\nStats: {json.dumps(bot.get_stats(), indent=2)}")
    print("\nSmoke test complete!")