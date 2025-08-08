import pytest
import sys

if __name__ == "__main__":
    sys.exit(pytest.main(["tests", "--asyncio-mode=auto"]))
