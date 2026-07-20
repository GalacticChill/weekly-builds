"""Collect a sample of GitHub repositories and their metadata.

We want an honest cross-section of repos — not just famous ones — so we sample
across several **star buckets** and languages. Sampling only popular repos would
be survivorship bias: we'd never see the thousands of polished-looking projects
that got no traction, which are exactly the counter-examples this study needs.

Auth uses your existing `gh` login (`gh auth token`) or a `GITHUB_TOKEN` env var.
The result is saved to CSV so the rest of the project runs offline and
reproducibly, without hitting the API again.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://api.github.com/search/repositories"

# (label, star query fragment). Buckets span from "basically ignored" to "famous"
# so the model sees the full range of outcomes.
STAR_BUCKETS = [
    ("tiny", "stars:1..9"),
    ("small", "stars:10..49"),
    ("medium", "stars:50..299"),
    ("large", "stars:300..1999"),
    ("huge", "stars:2000..19999"),
    ("mega", "stars:>=20000"),
]

LANGUAGES = ["Python", "JavaScript", "Go", "Rust"]

RAW_FIELDS = [
    "full_name",
    "stargazers_count",
    "forks_count",
    "open_issues_count",
    "size",
    "language",
    "description",
    "homepage",
    "has_wiki",
    "created_at",
    "pushed_at",
]


def _token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    try:
        return subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "No GitHub token found. Set GITHUB_TOKEN or run `gh auth login`."
        ) from exc


def _extract(item: dict) -> dict:
    row = {field: item.get(field) for field in RAW_FIELDS}
    row["topics"] = ";".join(item.get("topics") or [])
    lic = item.get("license")
    row["has_license"] = bool(lic and lic.get("key") and lic["key"] != "other")
    return row


def collect(
    per_query: int = 100,
    languages: list[str] | None = None,
    pause: float = 2.0,
) -> pd.DataFrame:
    """Fetch up to `per_query` repos for each (language, star-bucket) pair."""
    languages = languages or LANGUAGES
    headers = {
        "Authorization": f"Bearer {_token()}",
        "Accept": "application/vnd.github+json",
    }
    rows: list[dict] = []
    for lang in languages:
        for _, star_q in STAR_BUCKETS:
            query = f"{star_q} language:{lang} sort:stars"
            page, got = 1, 0
            while got < per_query:
                resp = requests.get(
                    API,
                    headers=headers,
                    params={"q": query, "per_page": min(100, per_query - got), "page": page},
                    timeout=30,
                )
                if resp.status_code == 403:  # secondary rate limit; back off
                    time.sleep(30)
                    continue
                resp.raise_for_status()
                items = resp.json().get("items", [])
                if not items:
                    break
                rows.extend(_extract(it) for it in items)
                got += len(items)
                page += 1
                time.sleep(pause)  # stay under the 30 req/min search limit
    df = pd.DataFrame(rows).drop_duplicates(subset="full_name").reset_index(drop=True)
    return df


def save(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
