from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests

from freebird.config import (
    VICOHOME_EMAIL,
    VICOHOME_PASSWORD,
    get_api_base,
)

logger = logging.getLogger(__name__)

AUTH_ERROR_CODES = {-1024, -1025, -1026, -1027}
TOKEN_TTL_SECONDS = 23 * 60 * 60  # 23h (conservative vs 24h server-side)
CACHE_PATH = Path.home() / ".freebird" / "auth.json"


class AuthManager:
    def __init__(self) -> None:
        self._token: str | None = None
        self._expires_at: float = 0
        self._load_cached_token()

    def get_token(self) -> str:
        if self._token and time.time() < self._expires_at:
            return self._token
        return self._login()

    def invalidate(self) -> None:
        self._token = None
        self._expires_at = 0
        if CACHE_PATH.exists():
            CACHE_PATH.unlink()

    def _login(self) -> str:
        logger.info("Logging in to VicoHome API")
        resp = requests.post(
            f"{get_api_base()}/account/login",
            json={
                "email": VICOHOME_EMAIL,
                "password": VICOHOME_PASSWORD,
                "loginType": 0,
            },
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        body = resp.json()

        result = body.get("result", body.get("code", -1))
        if result != 0:
            raise RuntimeError(f"VicoHome login failed: {body.get('msg', body)}")

        token = body["data"]["token"]["token"]
        self._token = token
        self._expires_at = time.time() + TOKEN_TTL_SECONDS
        self._save_cached_token()
        logger.info("Login successful, token cached")
        return token

    def _load_cached_token(self) -> None:
        if not CACHE_PATH.exists():
            return
        try:
            data = json.loads(CACHE_PATH.read_text())
            if data.get("expires_at", 0) > time.time():
                self._token = data["token"]
                self._expires_at = data["expires_at"]
                logger.info("Loaded cached token (expires in %.0f min)",
                            (self._expires_at - time.time()) / 60)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Invalid cached token, will re-login")

    def _save_cached_token(self) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        CACHE_PATH.write_text(json.dumps({
            "token": self._token,
            "expires_at": self._expires_at,
        }))
        CACHE_PATH.chmod(0o600)

    @staticmethod
    def is_auth_error(response_body: dict) -> bool:
        code = response_body.get("result", response_body.get("code"))
        if isinstance(code, int) and code in AUTH_ERROR_CODES:
            return True
        msg = str(response_body.get("msg", "")).lower()
        return any(kw in msg for kw in ("token", "auth", "login"))
