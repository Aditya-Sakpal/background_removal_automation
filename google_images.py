"""
Google Custom Search JSON API — image search for reference pool.

Requires in environment:
  GOOGLE_CSE_API_KEY — API key with Custom Search API enabled
  GOOGLE_CSE_ID — Programmable Search Engine ID (cx)
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

# Match gallery banner ~ 1176 x 516
DEFAULT_TARGET_ASPECT = 1176 / 516
REFERENCE_IMAGE_POOL_SIZE = 20

# Name first, then queries tuned for live event / action imagery.
QUERY_VARIANT_SUFFIXES = [
    "",  # bare name only — handled first in query_plan
    "microphone speaking live stage",
    "audience interaction event candid",
    "expressive performance concert live",
    "keynote speech podium crowd",
    "candid laughing talking event",
    "live show fan interaction stage",
]


def _aspect_ratio(width: int | None, height: int | None) -> float | None:
    if not width or not height or height == 0:
        return None
    return width / height


def _aspect_score(width: int | None, height: int | None, target: float) -> float:
    r = _aspect_ratio(width, height)
    if r is None:
        return -999.0
    return -abs(r - target)


def _name_tokens(name: str) -> set[str]:
    parts = re.sub(r"[^a-zA-Z0-9]+", " ", name.lower()).split()
    return {p for p in parts if len(p) >= 2}


def _title_matches_name(title: str, tokens: set[str]) -> bool:
    if not tokens:
        return False
    t = (title or "").lower()
    return any(tok in t for tok in tokens)


def _search_once(
    query: str,
    api_key: str,
    cx: str,
    *,
    start: int = 1,
) -> list[dict[str, Any]]:
    params: dict[str, str | int] = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "searchType": "image",
        "num": 10,
        "start": min(max(1, start), 91),
        "safe": "active",
        "gl": "in",
        "hl": "en",
        "imgType": "photo",
    }
    # Keep first query broad; for suffix queries bias to wide images.
    if query.strip() and " " in query.strip():
        params["imgSize"] = "LARGE"
        params["imgDominantColor"] = "black"
        params["rights"] = "cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived"

    url = f"{GOOGLE_CSE_ENDPOINT}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "engage4more-image-tool/1.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if "error" in payload:
        err = payload["error"]
        msg = err.get("message", str(err))
        raise RuntimeError(f"Google API error: {msg}")
    return list(payload.get("items") or [])


def _normalize_item(raw: dict[str, Any], target_aspect: float) -> dict[str, Any] | None:
    link = raw.get("link")
    if not link or not isinstance(link, str):
        return None
    img_meta = raw.get("image") or {}
    w = img_meta.get("width")
    h = img_meta.get("height")
    try:
        wi = int(w) if w is not None else None
        hi = int(h) if h is not None else None
    except (TypeError, ValueError):
        wi, hi = None, None
    thumb = img_meta.get("thumbnailLink")
    title = (raw.get("title") or "")[:200]
    return {
        "link": link,
        "title": title,
        "width": wi,
        "height": hi,
        "thumbnail_link": thumb if isinstance(thumb, str) else None,
        "aspect_ratio": _aspect_ratio(wi, hi),
        "aspect_score": _aspect_score(wi, hi, target_aspect),
        "display_url": link,
    }


def fetch_google_image_candidates(
    celebrity_name: str,
    api_key: str,
    cx: str,
    *,
    target_aspect: float = DEFAULT_TARGET_ASPECT,
    target_count: int = REFERENCE_IMAGE_POOL_SIZE,
) -> list[dict[str, Any]]:
    name = celebrity_name.strip()
    if not name:
        return []

    tokens = _name_tokens(name)
    seen_links: set[str] = set()
    pool: list[dict[str, Any]] = []

    query_plan: list[str] = [name]
    for suffix in QUERY_VARIANT_SUFFIXES[1:]:
        query_plan.append(f"{name} {suffix}".strip())

    # Pull first pages for all queries.
    for q in query_plan:
        if len(pool) >= target_count:
            break
        try:
            items = _search_once(q, api_key, cx, start=1)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise RuntimeError(f"Google HTTP {e.code}: {body[:500]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Google request failed: {e}") from e

        for raw in items:
            if len(pool) >= target_count:
                break
            norm = _normalize_item(raw, target_aspect)
            if not norm or norm["link"] in seen_links:
                continue
            # Prefer title matching first pass.
            if tokens and not _title_matches_name(norm["title"], tokens):
                continue
            r = norm.get("aspect_ratio")
            if r is not None and r < 1.25:
                continue
            seen_links.add(norm["link"])
            pool.append(norm)

    # Fallback without title gate.
    if len(pool) < target_count:
        for q in query_plan:
            if len(pool) >= target_count:
                break
            try:
                items = _search_once(q, api_key, cx, start=1)
            except (urllib.error.HTTPError, urllib.error.URLError):
                continue
            for raw in items:
                if len(pool) >= target_count:
                    break
                norm = _normalize_item(raw, target_aspect)
                if not norm or norm["link"] in seen_links:
                    continue
                r = norm.get("aspect_ratio")
                if r is not None and r < 1.25:
                    continue
                seen_links.add(norm["link"])
                pool.append(norm)

    # Pull second pages when still short.
    if len(pool) < target_count:
        for start in (11, 21):
            if len(pool) >= target_count:
                break
            for q in query_plan:
                if len(pool) >= target_count:
                    break
                try:
                    items = _search_once(q, api_key, cx, start=start)
                except (urllib.error.HTTPError, urllib.error.URLError):
                    continue
                for raw in items:
                    if len(pool) >= target_count:
                        break
                    norm = _normalize_item(raw, target_aspect)
                    if not norm or norm["link"] in seen_links:
                        continue
                    r = norm.get("aspect_ratio")
                    if r is not None and r < 1.25:
                        continue
                    seen_links.add(norm["link"])
                    pool.append(norm)

    return pool[:target_count]


def download_image_bytes(url: str, *, timeout: int = 25) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; engage4more-image-tool/1.0)",
            "Accept": "image/*,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()
