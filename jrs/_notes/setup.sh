#!/bin/bash

# This script installs LightRAG and RAGAnywhere into a directory named LightRAG


# The following checklist is for me only.
# It ensures I am giving you my latest code and the latest code from LightRAG too.
# ==============================================================================
# PRE-MIGRATION CHECKLIST (Run these on your OLD computer first!)
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
# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting RAGAnywhere Environment Setup..."

# 1. Install uv (Python package manager) if not present
if ! command -v uv &> /dev/null
then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the cargo env to make uv available immediately in this session
    source $HOME/.cargo/env
else
    echo "✅ uv is already installed."
fi

# 2. Clone the repository
REPO_URL="https://github.com/johnshearing/LightRAG.git"
REPO_DIR="LightRAG"

if [ ! -d "$REPO_DIR" ]; then
    echo "📂 Cloning your LightRAG fork..."
    git clone "$REPO_URL"
    cd "$REPO_DIR"
else
    echo "🏠 Directory $REPO_DIR already exists. Entering directory..."
    cd "$REPO_DIR"
    git pull origin main
fi

# 3. Create Virtual Environment and Sync Dependencies
echo "⚙️ Syncing dependencies with uv..."
# This installs core + api + offline + evaluation dependencies found in your pyproject.toml
uv sync --all-extras

# 4. Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  WARNING: No .env file detected."
    echo "   If you have a .env.example, run: cp .env.example .env"
    echo "   Then add your API keys to the .env file."
    echo "   Or for better security, add API keys to your .bashrc file instead."    
    echo "   No matter where you decide to put your API keys, you will still need a .env file for other required settings"    
fi

# 4. Final Verification
echo "-----------------------------------------------"
echo "🎉 Setup Complete!"
echo "To activate your environment, run: source .venv/bin/activate"
echo "To run your scripts: python jrs/your_script.py"
echo "-----------------------------------------------"