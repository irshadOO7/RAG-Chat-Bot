"""
cli.py - Command-line interface for the RAG Chat Bot

Provides an interactive REPL to upload documents, ask questions,
and manage the vector store.

Usage:
    python cli.py                              # Use Ollama with llama3
    python cli.py --backend transformers \
        --model google/flan-t5-base            # Use HuggingFace transformers
    python cli.py --model gemma2               # Use a different Ollama model
    python cli.py --help                       # Show all options
"""

import argparse
import sys
from rag_bot import RAGBot


def print_help():
    print("""\
RAG Chat Bot - Commands:
  /upload <path>     Upload a document (PDF, TXT, DOCX, MD)
  /text <content>    Add raw text
  /query <question>  Ask a question
  /clear             Clear all documents
  /stats             Show bot statistics
  /save <path>       Save vector store to disk
  /load <path>       Load vector store from disk
  /backend <name>    Switch LLM backend (ollama/transformers)
  /model <name>      Switch LLM model
  /help              Show this help message
  /quit              Exit

Just type your question directly (without /) to chat.
""")


def main():
    parser = argparse.ArgumentParser(description="RAG Chat Bot CLI")
    parser.add_argument("--backend", default="ollama",
                        choices=["ollama", "transformers"],
                        help="LLM backend (default: ollama)")
    parser.add_argument("--model", default="llama3",
                        help="LLM model name (default: llama3)")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Chunk size in tokens (default: 512)")
    parser.add_argument("--overlap", type=int, default=50,
                        help="Chunk overlap in tokens (default: 50)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="LLM temperature (default: 0.3)")
    args = parser.parse_args()

    print("=" * 50)
    print("  RAG Chat Bot - Local & Free")
    print("=" * 50)
    print(f"  Backend: {args.backend}")
    print(f"  Model:   {args.model}")
    print(f"  Top-K:   {args.top_k}")
    print("=" * 50)

    try:
        bot = RAGBot(
            llm_backend=args.backend,
            llm_model=args.model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print("\nBot ready! Type /help for commands.")
    except Exception as e:
        print(f"\nError initializing bot: {e}")
        sys.exit(1)

    print_help()

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(None, 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/quit":
                print("Goodbye!")
                break
            elif cmd == "/help":
                print_help()
            elif cmd == "/upload":
                if not arg:
                    print("Usage: /upload <path>")
                else:
                    try:
                        result = bot.add_document(arg)
                        print(f"Loaded '{result['source']}': "
                              f"{result['num_chunks']} chunks, "
                              f"{result['num_chars']} chars")
                    except Exception as e:
                        print(f"Error: {e}")
            elif cmd == "/text":
                if not arg:
                    print("Usage: /text <content>")
                else:
                    result = bot.add_text(arg, "raw_text")
                    print(f"Added text: {result['num_chunks']} chunks")
            elif cmd == "/clear":
                bot.clear()
            elif cmd == "/stats":
                stats = bot.get_stats()
                for k, v in stats.items():
                    print(f"  {k}: {v}")
            elif cmd == "/save":
                if not arg:
                    print("Usage: /save <path>")
                else:
                    try:
                        bot.save(arg)
                        print(f"Saved to {arg}")
                    except Exception as e:
                        print(f"Error: {e}")
            elif cmd == "/load":
                if not arg:
                    print("Usage: /load <path>")
                else:
                    try:
                        bot.load(arg)
                        print(f"Loaded from {arg}")
                    except Exception as e:
                        print(f"Error: {e}")
            elif cmd == "/backend":
                if not arg:
                    print("Usage: /backend <ollama|transformers>")
                else:
                    bot.switch_llm(arg, bot.llm_model)
            elif cmd == "/model":
                if not arg:
                    print("Usage: /model <model_name>")
                else:
                    bot.switch_llm(bot.llm_backend, arg)
            else:
                print(f"Unknown command: {cmd}. Type /help for commands.")
        else:
            try:
                result = bot.query(user_input)
                print(f"\n{result.answer}")
                if result.sources:
                    print(f"  Sources: {', '.join(result.sources)}")
                if result.retrieved_chunks:
                    best = result.retrieved_chunks[0]
                    print(f"  Best match score: {best[1]:.3f}")
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
