#!/bin/bash

# This script installs LightRAG and RAGAnywhere into a directory named LightRAG


# The following pre-migration checklist is for me only.
# It ensures I am giving you my latest code and the latest code from LightRAG too.
# ==============================================================================
# PRE-MIGRATION CHECKLIST (Run these on the computer with original source!)
# ==============================================================================
# 1. cd ~/LightRAG
# 2. git add .
# 3. git commit -m "Checkpoint: Save my work before syncing"
# 4. git fetch upstream
# 5. git merge upstream/main
#    --> If conflicts occur, fix them in .gitignore / pyproject.toml
# 6. uv lock
#    --> This ensures the lock file is healthy after the merge
# 7. git add .
# 8. git commit -m "Final sync: Integrated upstream changes"
#    --> (Note: Only required if Step 5 had conflicts)
# 9. git push origin main
# 10. git status
#     --> I should see: "Your branch is up to date with 'origin/main'" and "nothing to commit, working tree clean"
# ==============================================================================



#*******************Start of Notes*********************
# Move this file to your home directory before running it.
# This will cause the LightRAG directory to be created in your home directory.

# !!!!!!!!!!!!!!!!     Important!      !!!!!!!!!!!!!!!!!
# If you already have a directory named LightRAG then name it something else so that it will not be overwritten.
# run the following bash command:
# mv ~/LightRAG ~/LightRAG_BACKUP

# This script here that you are reading will not run if it is not marked by the OS as an executable file.
# Make this file executable with the following bash command in your terminal window:
# chmod +x setup.sh

# Then run the script with the following bash command in your terminal window:
# ./setup.sh

# LightRAG and RAGAnywhere require environment variables in order to run. (API keys, etc.) 
# Ensure API keys are defined in a .env file in the LightRAG directory.
# For better security, define API keys in the .bashrc file in you home directory rather than in your .env file.
# The following line should be in your .env file, or for better security in your .bashrc file.
#export OPENAI_API_KEY="My_API_Key"
#*******************End of Notes*********************





#*******************Start of Script******************
set -e

echo "🚀 Starting RAGAnywhere Environment Setup..."

# 1. Install uv (Python package manager)
if ! command -v uv &> /dev/null
then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
else
    echo "✅ uv is already installed."
fi

# 2. Install bun (Needed for Web UI)
if ! command -v bun &> /dev/null
then
    echo "📦 Installing bun..."
    curl -fsSL https://bun.sh/install | bash
    # For a script, we source the specific bin path rather than the whole bashrc
    export PATH="$HOME/.bun/bin:$PATH"
else
    echo "✅ bun is already installed."
fi

# 3. Clone the repository
REPO_URL="https://github.com/johnshearing/LightRAG.git"
REPO_DIR="LightRAG"

if [ ! -d "$REPO_DIR" ]; then
    echo "📂 Cloning your LightRAG fork..."
    git clone "$REPO_URL"
    cd "$REPO_DIR"
else
    echo "🏠 Directory $REPO_DIR already exists. Updating..."
    cd "$REPO_DIR"
    git pull origin main
fi

# 4. Create Virtual Environment and Sync Dependencies
echo "⚙️ Syncing Python dependencies with uv..."
uv sync --all-extras

echo "📦 Installing additional RAGAnything components..."
uv pip install raganything

# 5. Build the Web UI
echo "🌐 Building Web UI..."
if [ -d "lightrag_webui" ]; then
    cd lightrag_webui
    bun install --frozen-lockfile
    bun run build
    cd ..
    echo "✅ Web UI built successfully."
else
    echo "❌ Error: lightrag_webui directory not found!"
    exit 1
fi

echo "-----------------------------------------------"
echo "🎉 Setup Complete!"

# 6. Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  WARNING: No .env file detected."
    echo "   Action: cp env.example .env"
    echo "   Then add your API keys to the .env file."
    echo "   Or for better security, add API keys to your .bashrc file instead."    
    echo "   No matter where you decide to put your API keys, you will still need a .env file for other required settings" 
    echo "   If you prefer, you can use the .env file I have in the "LightRAG/jrs/_notes" directory which is working for me."
    echo "   Just copy that file to the LightRAG folder."            
fi


# 7. Final Verification and Instructions
cat << EOF


To use LightRAG, run these commands:
  1. cd LightRAG

  2. source .venv/bin/activate
  This activates the virtual environment.

  3. export PS1='(.venv) \w\$ '
  This creates a better looking prompt that takes up less space on the screen.

  4. lightrag-server
  This starts the server.

  5. Visit: http://localhost:9621/webui/ 
  This WebUI is where you interact with the server.

  6. You can interact directly with LightRAG using the
     Python scripts in the jrs directory without the need
     for the lightrag server or the WebUI 
-----------------------------------------------
EOF