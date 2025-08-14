from typing import Any, Dict
from .http import AsyncHTTP
from .config import Settings

async def api_health(http: AsyncHTTP, s: Settings) -> Dict[str, Any]:
    url = s.health_url
    resp = await http.get(url)
    ct = resp.headers.get("content-type","")
    if "json" in ct:
        try:
            payload = resp.json()
        except Exception:
            payload = {"status_text": resp.text}
    else:
        payload = {"status_text": resp.text}
    payload.setdefault("status", payload.get("status_text","unknown"))
    payload["_status_code"] = resp.status_code
    payload["_ok"] = resp.status_code == 200
    return payload
