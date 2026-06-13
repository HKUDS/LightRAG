"""Start LightRAG server with LIGHTRAG_FRAME_EXTRACTION_MODE=llm_frames.

Wraps the standard server startup, ensuring the frame extraction mode is
set correctly and Windows console encoding errors are suppressed.
"""
import os
import sys
import io

# Force frame extraction mode before any LightRAG imports
os.environ["LIGHTRAG_FRAME_EXTRACTION_MODE"] = "llm_frames"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

# Wrap stdout/stderr so that UnicodeEncodeError doesn't crash the server.
# ASCIIColors writes emoji/box-drawing chars that fail on cp1252 terminals.
class _SafeWriter(io.TextIOWrapper):
    def write(self, s):
        try:
            return super().write(s)
        except (UnicodeEncodeError, UnicodeDecodeError):
            return super().write(s.encode("ascii", errors="replace").decode("ascii"))

    def flush(self):
        try:
            super().flush()
        except Exception:
            pass

# Only wrap if stdout is file-backed (i.e. redirected, not a real console)
if hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = _SafeWriter(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass
if hasattr(sys.stderr, "buffer"):
    try:
        sys.stderr = _SafeWriter(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

print(f"[start] LIGHTRAG_FRAME_EXTRACTION_MODE = {os.environ['LIGHTRAG_FRAME_EXTRACTION_MODE']}", flush=True)

from lightrag.api.lightrag_server import main
main()
