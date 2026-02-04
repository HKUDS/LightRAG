#!/bin/bash

# This script installs a development version of LightRAG and RAGAnywhere into a directory named LightRAG-Dev.
# This installation will allow modifications to the WebUI


# The following pre-migration checklist is for me only.
# It ensures I am giving you my latest code and the latest code from LightRAG too.
# ==============================================================================
# PRE-MIGRATION & SYNC CHECKLIST (Run this in your Dev environment)
# ==============================================================================
# 1.  --- Navigate to the directory we wish to sync ---
    #   cd ~/LightRAG-Dev

# 2.  --- SAVE YOUR WEBUI WORK ---
#     Ensure your frontend changes are built and staged
    #   cd lightrag_webui
    #   bun run build
    #   cd ..

# 3.  --- COMMIT LOCAL CHANGES ---
    #   git add .
    #   git commit -m "Checkpoint: Save WebUI modifications and built assets"

# 4.  --- CREATE CHECKPOINT FOR DISASTER ROLLBACK --
    # git branch backup/pre-migration-2026-02-02   !!! USE THE CORRECT DATE !!!
    # git push origin backup/pre-migration-2026-02-02

# 5.  --- SYNC WITH UPSTREAM (HKUDS/LightRAG) ---
    #   git fetch upstream
    #   git merge upstream/main
#     --> Note: If conflicts occur, prioritize keeping your new WebUI files
#         but accept upstream fixes for the core RAG logic.

# 6.  --- UPDATE DEPENDENCY LOCK FILES ---
#     Ensure both Python and Frontend dependencies are healthy after the merge
    #   uv lock
    #   cd lightrag_webui
    #   bun install
    #   cd ..

# 7.  --- OPTIONAL AND UNTESTED: PURGE GHOST ARTIFACTS ---
# Clear Python/uv cache
    # uv cache clean

# Navigate to WebUI and clear Vite/Bun cache
    # cd lightrag_webui

# Remove the local 'dist' and Vite's internal cache
    # rm -rf dist node_modules/.vite
# (Optional) Re-run build to ensure everything is fresh
    # bun run build
    # cd ..

# 8.  --- FINAL COMMIT ---
    #   git add .
    #   git commit -m "Final sync: Integrated upstream changes & updated lock files"
#     --> (Only required if Step 4 had conflicts or Step 5 updated locks)

# 9.  --- PUSH TO YOUR FORK ---
    #   git push origin main

# 10. --- VERIFY ---
    #   git status
#     --> Should see: "Your branch is up to date with 'origin/main'"
# ==============================================================================




# ==============================================================================
# INCASE OF DISASTER WITH THE ABOVE SYNC OPERATION, RECOVER WITH:
# git checkout main
# git reset --hard backup/pre-migration-2026-02-02
# ==============================================================================



#*******************Start of Notes*********************
# Move this file to your home directory before running it.
# This will cause the LightRAG-Dev directory to be created in your home directory.

# !!!!!!!!!!!!!!!!     Important!      !!!!!!!!!!!!!!!!!
# If you already have a directory named LightRAG-Dev then change the
# REPO_DIR variable in this script to something other than LightRAG-Dev so that
# your orgininal work will not be overwritten.

# This script here that you are reading will not run if it is not marked by the OS as an executable file.
# Make this file executable with the following bash command in your terminal window:
# chmod +x setup-dev.sh

# Then run the script with the following bash command in your terminal window:
# ./setup-dev.sh

# LightRAG and RAGAnywhere require environment variables in order to run. (API keys, etc.)
# Ensure API keys are defined in a .env file in the LightRAG directory.
# For better security, define API keys in the .bashrc file in you home directory rather than in your .env file.
# The following line should be in your .env file, or for better security in your .bashrc file.
#export OPENAI_API_KEY="My_API_Key"
#*******************End of Notes*********************





#*******************Start of Script******************
set -e

echo "🚀 Starting LightRAG WebUI Development Setup..."

# 1. Install uv (Python package manager)
if ! command -v uv &> /dev/null
then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
else
    echo "✅ uv is already installed."
fi

# 2. Install bun (Required by LightRAG for Web UI)
if ! command -v bun &> /dev/null
then
    echo "📦 Installing bun..."
    curl -fsSL https://bun.sh/install | bash
    export PATH="$HOME/.bun/bin:$PATH"
else
    echo "✅ bun is already installed."
fi

# 3. Clone the repository
REPO_URL="https://github.com/johnshearing/LightRAG.git"
REPO_DIR="LightRAG-Dev"

if [ ! -d "$REPO_DIR" ]; then
    echo "📂 Cloning your LightRAG fork for development..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
    git remote add upstream https://github.com/HKUDS/LightRAG.git
else
    echo "🏠 Directory $REPO_DIR already exists. Updating..."
    cd "$REPO_DIR"
    git pull origin main
fi

# 4. Create Virtual Environment and Sync for DEVELOPMENT
echo "⚙️ Setting up Python environment in EDITABLE mode..."
# Using --editable ensures changes to the source code are live
uv sync --all-extras --editable

echo "📦 Installing additional RAGAnything components..."
uv pip install raganything

# 5. Build the Web UI
echo "🌐 Setting up Web UI for development..."
if [ -d "lightrag_webui" ]; then
    cd lightrag_webui
    echo "Installing frontend dependencies with bun..."
    bun install --frozen-lockfile

    echo "Performing initial frontend build..."
    # This places the initial assets into lightrag/api/webui/
    bun run build
    cd ..
    echo "✅ Web UI initialized."
else
    echo "❌ Error: lightrag_webui directory not found!"
    exit 1
fi

echo "-----------------------------------------------"
echo "🎉 Setup Complete!"

# 6. Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  WARNING: No .env file detected."
    echo "   Please copy your working .env from your backup or jrs/_notes folder."
    echo "   Then add your API keys to the .env file."
    echo "   Or for better security, add API keys to your .bashrc file instead."
    echo "   No matter where you decide to put your API keys, you will still need a .env file for other required settings"
fi


# 7. Verification and Instructions
cat << EOF

To DEVELOP and MODIFY the WebUI, follow these steps:

1. ACTIVATE ENVIRONMENT:
   cd $REPO_DIR
   source .venv/bin/activate

2. CREATE A PROMPT THAT TAKES UP LESS SPACE.
   export PS1='(.venv) \w\$ '

2. START THE BACKEND:
   lightrag-server

3. LIVE UI DEVELOPMENT (Hot Reloading):
   In a NEW terminal:
   cd $REPO_DIR/lightrag_webui
   bun run dev
   # This will give you a link (usually http://localhost:5173)
   # that updates instantly as you change code.

4. FINALIZING UI CHANGES:
   When your changes are ready, run:
   cd $REPO_DIR/lightrag_webui
   bun run build
   # This updates the files served by the main lightrag-server.

-----------------------------------------------
EOF
