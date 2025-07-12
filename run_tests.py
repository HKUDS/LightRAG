#!/usr/bin/env python3
"""
LightRAG Test Suite - Quick launcher
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.test_cli import main

if __name__ == "__main__":
    main()
