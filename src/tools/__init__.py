"""LangChain tool modules for the games bot agent."""

from src.tools.store import ALL_TOOLS as STORE_TOOLS
from src.tools.web import ALL_TOOLS as WEB_TOOLS
from src.tools.media import ALL_TOOLS as MEDIA_TOOLS
from src.tools.igdb import ALL_TOOLS as IGDB_TOOLS

PYTHON_TOOLS = STORE_TOOLS + WEB_TOOLS + MEDIA_TOOLS + IGDB_TOOLS
