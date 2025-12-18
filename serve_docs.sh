#!/bin/bash
# Serve feature documentation locally

DOCS_DIR="docs/features"
PORT=8000

echo "🚀 Starting documentation server..."
echo "📂 Serving from: $DOCS_DIR"
echo "🌐 URL: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd "$DOCS_DIR" && python3 -m http.server $PORT
