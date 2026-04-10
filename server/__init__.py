"""MegaGem multiplayer server.

FastAPI + WebSocket front door for the canonical Python engine that
lives under ``research/megagem``. Importing anything from ``server``
transparently adds ``research/`` to ``sys.path`` so
``from megagem import ...`` resolves against the research tree without
having to install it as a package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_RESEARCH_DIR = Path(__file__).resolve().parent.parent / "research"
if str(_RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_DIR))
