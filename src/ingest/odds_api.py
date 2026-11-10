"""Minimal client for The Odds API (v4) + key loading.

The key is read from the ODDS_API_KEY environment variable, or from a file called `.env`
in the project root containing a line:   ODDS_API_KEY=your_key_here
The key is never printed or written to any report.
"""
from __future__ import annotations
import os
from pathlib import Path
import requests
from src.config import ROOT

BASE = "https://api.the-odds-api.com/v4"


PLACEHOLDERS = ("paste", "your_key", "your-key", "api_key_here", "xxxx")


def _from_streamlit_secrets(name: str) -> str | None:
    """Streamlit Cloud 'Secrets' (also works locally with .streamlit/secrets.toml). Never raises."""
    try:
        import streamlit as st
        v = st.secrets.get(name)
        return str(v).strip() if v else None
    except Exception:  # noqa: BLE001  (no secrets configured / streamlit not installed)
        return None


def find_secret(name: str, env_path: Path | None = None, required: bool = True) -> tuple[str | None, str]:
    """Return (value, where it came from). Order: environment variable, .env file, Streamlit secrets."""
    val = os.environ.get(name)
    if val and val.strip():
        return val.strip(), f"the {name} environment variable"
    p = Path(env_path) if env_path else ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == name:
                v = v.strip().strip(chr(34)).strip(chr(39))
                if v:
                    return v, f"the .env file at {p}"
    v = _from_streamlit_secrets(name)
    if v:
        return v, "Streamlit secrets"
    if required:
        raise SystemExit(
            f"{name} not found.\n"
            f"Expected a file named exactly .env in {ROOT} containing one line:\n"
            f"{name}=your_key_here\n"
            "(Check it is not saved as .env.txt.)")
    return None, ""


def find_key(env_path: Path | None = None) -> tuple[str, str]:
    return find_secret("ODDS_API_KEY", env_path)


def load_key(env_path: Path | None = None) -> str:
    return find_key(env_path)[0]


def key_warnings(key: str) -> list[str]:
    out = []
    if any(w in key.lower() for w in PLACEHOLDERS):
        out.append("The key still looks like the placeholder text from .env.example.")
    if len(key) < 20:
        out.append(f"The key is only {len(key)} characters long; real keys are longer. Was it cut off when pasting?")
    if any(ch.isspace() for ch in key) or any(ch in key for ch in chr(34) + chr(39)):
        out.append("The key contains spaces or quote characters.")
    return out


class OddsClient:
    def __init__(self, key: str):
        self._key = key
        self.remaining = None
        self.used = None

    def get(self, path: str, **params) -> requests.Response:
        params["apiKey"] = self._key
        r = requests.get(f"{BASE}{path}", params=params, timeout=30)
        self.remaining = r.headers.get("x-requests-remaining", self.remaining)
        self.used = r.headers.get("x-requests-used", self.used)
        return r
