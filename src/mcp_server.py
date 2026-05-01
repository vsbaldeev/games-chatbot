"""
MCP server exposing game research tools.
Run as a subprocess via stdio transport: python src/mcp_server.py

IMPORTANT: Never print() to stdout — it breaks the MCP JSON-RPC channel.
           Use logging to stderr only.
"""

import logging
import sys
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from src import config
from src.tools import igdb_tools, media_tools, store_tools, web_tools

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(server):
    try:
        await igdb_tools.get_igdb_token()
        logger.info("Twitch/IGDB authentication OK")
    except KeyError as error:
        logger.error("Missing environment variable %s — set TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET", error)
    except Exception as error:
        logger.error("Twitch startup check failed: %s: %s", type(error).__name__, error)
    yield


mcp = FastMCP("games-tools", lifespan=lifespan)

igdb_tools.register(mcp)
store_tools.register(mcp)
web_tools.register(mcp, config)
media_tools.register(mcp, config)

if __name__ == "__main__":
    mcp.run(transport="stdio")
