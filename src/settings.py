"""Paramètres partagés par les exemples d'agents (src/agent_*.py)."""

import os
from pathlib import Path

from dotenv import load_dotenv

# charge les variables du fichier .env à la racine du projet
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# coupe la bannière affichée par pydantic-ai au premier run
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

ENDPOINT = "https://api.scaleway.ai/dfe755c9-9fae-499b-9f6d-a0aad522acd7/v1"
MODEL = "deepseek-v4-flash-0731"
API_KEY = os.environ["GWENLAKE_API_KEY"]

# serveur MCP local utilisé par agent_08_mcp.py
MCP_SERVER_PATH = Path(__file__).resolve().parent / "mcp_server.py"
