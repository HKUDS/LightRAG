"""Minimal setup.py for backward compatibility with frontend build check"""
<<<<<<< HEAD

=======
>>>>>>> be9e6d16 (Exclude Frontend Build Artifacts from Git Repository)
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py
import sys


def check_webui_exists():
    """Check if WebUI has been built"""
    webui_index = Path("lightrag/api/webui/index.html")
    return webui_index.exists()


class BuildPyCommand(build_py):
    """Check WebUI build status before packaging/installation"""
<<<<<<< HEAD

    def run(self):
        # Check if running in development mode
        is_develop = any(arg in sys.argv for arg in ["develop", "egg_info"])
        is_editable = "--editable" in sys.argv or "-e" in sys.argv

=======
    
    def run(self):
        # Check if running in development mode
        is_develop = any(arg in sys.argv for arg in ['develop', 'egg_info'])
        is_editable = '--editable' in sys.argv or '-e' in sys.argv
        
>>>>>>> be9e6d16 (Exclude Frontend Build Artifacts from Git Repository)
        if is_develop or is_editable:
            # Development mode: friendly reminder
            if not check_webui_exists():
                print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  ℹ️  Development Mode - WebUI not built yet                             ║
╚══════════════════════════════════════════════════════════════════════════╝

You're installing in development mode. You can build the frontend later:

    cd lightrag_webui
    bun install
    bun run build

The changes will take effect immediately (symlink mode).
╚══════════════════════════════════════════════════════════════════════════╝
""")
        else:
            # Normal installation/packaging mode: frontend build required
            if not check_webui_exists():
                print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    ⚠️  ERROR: WebUI Not Built                            ║
╚══════════════════════════════════════════════════════════════════════════╝

For normal installation (pip install .), you must build the frontend first:

    cd lightrag_webui
    bun install
    bun run build
    cd ..

Then run the installation again.

💡 TIP: For development, use editable mode instead:
   pip install -e ".[api]"
<<<<<<< HEAD

=======
   
>>>>>>> be9e6d16 (Exclude Frontend Build Artifacts from Git Repository)
   This allows you to build the frontend after installation.

╚══════════════════════════════════════════════════════════════════════════╝
""")
                sys.exit(1)
<<<<<<< HEAD

=======
        
>>>>>>> be9e6d16 (Exclude Frontend Build Artifacts from Git Repository)
        print("✅ Proceeding with package build...")
        build_py.run(self)


setup(
    cmdclass={
<<<<<<< HEAD
        "build_py": BuildPyCommand,
=======
        'build_py': BuildPyCommand,
>>>>>>> be9e6d16 (Exclude Frontend Build Artifacts from Git Repository)
    }
)
