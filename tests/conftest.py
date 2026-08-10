"""Make the ``src`` layout importable when running ``pytest`` without install."""

import os
import sys

_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
_SRC = os.path.abspath(_SRC)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
