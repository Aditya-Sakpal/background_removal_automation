"""
SerpApi Google Images — reference pool for gallery-style wide images.

Requires:
  SERPAPI_API_KEY — from https://serpapi.com/manage-api-key

Docs: https://serpapi.com/google-images-api
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"

# Match gallery banner ~ 1176 x 516
DEFAULT_TARGET_ASPECT = 1176 / 516
REFERENCE_IMAGE_POOL_SIZE = 20

# Name first, then queries tuned for **live event / action** imagery (not studio headshots).
# Phrases mirror: speaking on stage, audience interaction, candid, expressive moments.
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


def _serpapi_google_images(
    api_key: str,
    *,
    q: str,
    ijn: int = 0,
    imgar: str | None = "w",
) -> dict[str, Any]:
    params: dict[str, str | int] = {
        "engine": "google_images",
        "q": q,
        "api_key": api_key,
        "ijn": ijn,
        "safe": "active",
        "google_domain": "google.co.in",
        "gl": "in",
        "hl": "en",
        "imgsz": "l",
    }
    if imgar:
        params["imgar"] = imgar
    url = f"{SERPAPI_SEARCH_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "engage4more-image-tool/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("error"):
        err = data["error"]
        msg = err if isinstance(err, str) else err.get("message", str(err))
        raise RuntimeError(f"SerpApi error: {msg}")
    return data


def _normalize_serp_item(raw: dict[str, Any], target_aspect: float) -> dict[str, Any] | None:
    link = raw.get("original") or raw.get("link")
    if not link or not isinstance(link, str):
        return None
    wi = raw.get("original_width")
    hi = raw.get("original_height")
    try:
        wi = int(wi) if wi is not None else None
        hi = int(hi) if hi is not None else None
    except (TypeError, ValueError):
        wi, hi = None, None
    thumb = raw.get("thumbnail")
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


def _name_tokens(name: str) -> set[str]:
    parts = re.sub(r"[^a-zA-Z0-9]+", " ", name.lower()).split()
    return {p for p in parts if len(p) >= 2}


def _title_matches_name(title: str, tokens: set[str]) -> bool:
    if not tokens:
        return False
    t = (title or "").lower()
    return any(tok in t for tok in tokens)


def fetch_serpapi_image_candidates(
    celebrity_name: str,
    api_key: str,
    *,
    target_aspect: float = DEFAULT_TARGET_ASPECT,
    target_count: int = REFERENCE_IMAGE_POOL_SIZE,
) -> list[dict[str, Any]]:
    """
    Fetch Google Images via SerpApi.

    Important: we keep **Google's relevance order** (SerpApi `images_results` order)
    and walk queries from strict (name-only) → light suffixes. We do **not**
    re-sort purely by aspect ratio — that was surfacing random wide banners
    unrelated to the celebrity.
    """
    name = celebrity_name.strip()
    if not name:
        return []

    tokens = _name_tokens(name)
    seen_links: set[str] = set()
    pool: list[dict[str, Any]] = []

    # (query_string, use_imgar_wide) — first query: no imgar so face/people results rank naturally
    query_plan: list[tuple[str, bool]] = [(name, False)]
    for suffix in QUERY_VARIANT_SUFFIXES[1:]:
        q = f"{name} {suffix}".strip()
        query_plan.append((q, True))

    for q, use_wide in query_plan:
        if len(pool) >= target_count:
            break
        try:
            payload = _serpapi_google_images(
                api_key,
                q=q,
                ijn=0,
                imgar="w" if use_wide else None,
            )
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise RuntimeError(f"SerpApi HTTP {e.code}: {body[:500]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"SerpApi request failed: {e}") from e

        for raw in payload.get("images_results") or []:
            if len(pool) >= target_count:
                break
            norm = _normalize_serp_item(raw, target_aspect)
            if not norm:
                continue
            if norm["link"] in seen_links:
                continue
            # Prefer images whose title mentions the person (reduces wrong faces)
            if tokens and not _title_matches_name(norm["title"], tokens):
                continue
            # Skip extreme portrait tiles (optional banner-friendly bias, soft)
            r = norm.get("aspect_ratio")
            if r is not None and r < 1.25:
                continue
            seen_links.add(norm["link"])
            pool.append(norm)

    # If title filter was too strict, fall back: fill by relevance order without title gate
    if len(pool) < target_count:
        for q, use_wide in query_plan:
            if len(pool) >= target_count:
                break
            try:
                payload = _serpapi_google_images(
                    api_key,
                    q=q,
                    ijn=0,
                    imgar="w" if use_wide else None,
                )
            except (urllib.error.HTTPError, urllib.error.URLError):
                continue
            for raw in payload.get("images_results") or []:
                if len(pool) >= target_count:
                    break
                norm = _normalize_serp_item(raw, target_aspect)
                if not norm or norm["link"] in seen_links:
                    continue
                r = norm.get("aspect_ratio")
                if r is not None and r < 1.25:
                    continue
                seen_links.add(norm["link"])
                pool.append(norm)

    # Extra pages (ijn 1–2) if we still need more unique images (SerpApi charges per request)
    if len(pool) < target_count:
        for ijn in (1, 2):
            if len(pool) >= target_count:
                break
            for q, use_wide in query_plan:
                if len(pool) >= target_count:
                    break
                try:
                    payload = _serpapi_google_images(
                        api_key,
                        q=q,
                        ijn=ijn,
                        imgar="w" if use_wide else None,
                    )
                except (urllib.error.HTTPError, urllib.error.URLError):
                    continue
                for raw in payload.get("images_results") or []:
                    if len(pool) >= target_count:
                        break
                    norm = _normalize_serp_item(raw, target_aspect)
                    if not norm or norm["link"] in seen_links:
                        continue
                    r = norm.get("aspect_ratio")
                    if r is not None and r < 1.25:
                        continue
                    seen_links.add(norm["link"])
                    pool.append(norm)

    return pool[:target_count]


def download_image_bytes(url: str, *, timeout: int = 25) -> bytes:
    """Download image bytes for display; raises on failure."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; engage4more-image-tool/1.0)",
            "Accept": "image/*,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()
