#!/bin/bash
# setup.sh - Automated setup for the RAG Chat Bot
# This script installs Python dependencies and optionally pulls an Ollama model.

set -e

echo "============================================"
echo "  RAG Chat Bot - Setup"
echo "============================================"

# --- Check Python ---
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    echo "Install from https://www.python.org/downloads/"
    exit 1
fi

echo "Python: $(python3 --version)"
echo ""

# --- Install Python dependencies ---
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# --- Check Ollama ---
echo ""
echo "Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed."
    echo "Install from: https://ollama.com/download"
    echo ""
    echo "After installing Ollama, run these commands:"
    echo "  ollama serve          # Start the Ollama server"
    echo "  ollama pull phi3    # Download the Llama 3 model"
else
    echo "Ollama found: $(ollama --version 2>/dev/null || echo 'installed')"
    echo ""
    echo "Starting Ollama server in background..."
    ollama serve &
    OLLAMA_PID=$!
    sleep 3

    echo "Pulling llama3 model (this may take a while)..."
    ollama pull phi3

    echo "Stopping background Ollama server..."
    kill $OLLAMA_PID 2>/dev/null || true
fi

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "To run the web app:   streamlit run app.py"
echo "To run the CLI:        python3 cli.py"
echo ""
