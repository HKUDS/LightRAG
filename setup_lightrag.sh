#!/bin/bash

set -e

# Check for pyenv
if ! command -v pyenv &> /dev/null; then
  echo "pyenv not found! Please install pyenv first."
  exit 1
fi

# 1. Set Python version (using pyenv)
PYTHON_VERSION=3.10.14

echo "Setting up Python $PYTHON_VERSION with pyenv..."
if ! pyenv versions | grep -q $PYTHON_VERSION; then
  pyenv install $PYTHON_VERSION
fi
pyenv local $PYTHON_VERSION

# 2. Create and activate venv
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python -m venv .venv
fi
source .venv/bin/activate

# 3. Upgrade pip and wheel
pip install --upgrade pip wheel

# 4. Install LightRAG core and API server in editable mode
pip install -e .
pip install -e ".[api]"

# 5. Copy .env if not present
if [ ! -f ".env" ]; then
  echo "Copying env.example to .env..."
  cp env.example .env
fi

# 6. Set Jina API key in .env
JINA_KEY="jina_403c398fc20d4a0da0cd46933e16622aCeXtW-cQVfAfiwXDd7vYnwYcbjsX"
if grep -q "^JINA_API_KEY=" .env; then
  sed -i '' "s|^JINA_API_KEY=.*|JINA_API_KEY=$JINA_KEY|" .env
else
  echo "JINA_API_KEY=$JINA_KEY" >> .env
fi

echo "JINA_API_KEY set to: $JINA_KEY"

echo "Setup complete!"
read -p "Do you want to activate the virtual environment now? (y/n): " yn
case $yn in
    [Yy]* ) source .venv/bin/activate;;
    * ) echo "You can activate it later with: source .venv/bin/activate";;
esac

read -p "Do you want to start the LightRAG server now? (y/n): " yn
case $yn in
    [Yy]* ) lightrag-server;;
    * ) echo "You can start it later with: lightrag-server";;
esac 