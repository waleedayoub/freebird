from __future__ import annotations

import logging
import time
from typing import Any

import requests

from freebird.config import get_api_base, get_country_no
from freebird.vicohome.auth import AuthManager
from freebird.vicohome.models import MotionEvent

logger = logging.getLogger(__name__)


class VicoHomeAPI:
    def __init__(self, auth: AuthManager | None = None) -> None:
        self.auth = auth or AuthManager()

    def _request(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Make an authenticated API request with auto-retry on auth errors."""
        base = get_api_base()
        url = f"{base}{endpoint}"

        for attempt in range(2):
            token = self.auth.get_token()
            resp = requests.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": token,
                },
                timeout=15,
            )
            resp.raise_for_status()

            # Check for HTML error responses (auth redirect)
            if resp.text.lstrip().startswith("<"):
                if attempt == 0:
                    logger.warning("Got HTML response, refreshing token")
                    self.auth.invalidate()
                    continue
                raise RuntimeError("VicoHome API returned HTML after token refresh")

            body = resp.json()

            if AuthManager.is_auth_error(body):
                if attempt == 0:
                    logger.warning("Auth error (code=%s), refreshing token",
                                   body.get("result", body.get("code")))
                    self.auth.invalidate()
                    continue
                raise RuntimeError(f"VicoHome auth failed after retry: {body.get('msg')}")

            return body

        raise RuntimeError("VicoHome API request failed after retries")

    def get_events(
        self,
        start_timestamp: int | None = None,
        end_timestamp: int | None = None,
    ) -> list[MotionEvent]:
        now = int(time.time())
        if end_timestamp is None:
            end_timestamp = now
        if start_timestamp is None:
            # Default: look back 1 hour
            start_timestamp = now - 3600

        payload = {
            "startTimestamp": str(start_timestamp),
            "endTimestamp": str(end_timestamp),
            "language": "en",
            "countryNo": get_country_no(),
        }
        body = self._request("/library/newselectlibrary", payload)

        code = body.get("code", body.get("result", -1))
        if code != 0:
            logger.error("Event list failed: %s", body.get("msg"))
            return []

        raw_list = body.get("data", {}).get("list", [])
        events = []
        for item in raw_list:
            try:
                events.append(MotionEvent.model_validate(item))
            except Exception:
                logger.warning("Failed to parse event: %s", item.get("traceId", "?"))
        return events

    def get_event(self, trace_id: str) -> MotionEvent | None:
        payload = {
            "traceId": trace_id,
            "language": "en",
            "countryNo": get_country_no(),
        }
        body = self._request("/library/newselectsinglelibrary", payload)

        code = body.get("code", body.get("result", -1))
        if code != 0:
            logger.error("Single event fetch failed: %s", body.get("msg"))
            return None

        data = body.get("data", {})
        # Response may have traceId at top level or nested under "event"
        if "traceId" not in data and "event" in data:
            data = data["event"]

        try:
            return MotionEvent.model_validate(data)
        except Exception:
            logger.warning("Failed to parse single event: %s", trace_id)
            return None
