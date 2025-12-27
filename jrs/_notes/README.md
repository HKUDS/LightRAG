# JRS Custom Scripts (RAGAnywhere)

This folder contains custom scripts for the LightRAG repository to implement RAGAnywhere functionality.

## 📂 Folder Structure
- `work/`: **(Ignored by Git)** This folder is used for temporary processing, logs, and local data. Do not put source code here.
- `*.py`: Main execution scripts for RAG processing.

## 🚀 How to Run
Ensure you are in the root of the LightRAG repository and your virtual environment is active:

1. **Install dependencies:**
   ```bash
   uv sync --all-extras

2. **Run a script:**
   ```bash
   python jrs/your_script_name.py


   📝 Configuration
If these scripts require environment variables (API keys, etc.), ensure they are defined in a .env file in the project root.

⚠️ Important Note
The work/ subdirectory is excluded from version control to prevent large local datasets from being pushed to GitHub. Always back up important data in work/ manually if needed.