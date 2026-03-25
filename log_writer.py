"""Read and write per-repo JSON log files and file-tree snapshots in the store/ directory."""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Resolve store/ relative to this file so imports from any working directory work correctly
_STORE_DIR = Path(__file__).parent / "store"


def _slugify(name: str) -> str:
    """Convert a repo display name to a safe filename slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    return slug


def _log_path(repo_name: str) -> Path:
    """Return the Path for a repo's JSON log file, creating the store dir if needed."""
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    return _STORE_DIR / f"{_slugify(repo_name)}.json"


def _tree_path(repo_name: str) -> Path:
    """Return the Path for a repo's file-tree JSON file."""
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    return _STORE_DIR / f"{_slugify(repo_name)}-tree.json"


def read_log(repo_name: str) -> list[dict]:
    """Load and return all log entries for the given repo, or an empty list if none exist."""
    path = _log_path(repo_name)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[log_writer] WARNING: Could not read log '{path}': {exc}", file=sys.stderr)
        return []


def append_entry(repo_name: str, entry_dict: dict) -> None:
    """Append a single summarised entry to the repo's JSON log file."""
    entries = read_log(repo_name)
    next_id = (entries[-1]["id"] + 1) if entries else 1
    entry_dict = {"id": next_id, **entry_dict}
    entries.append(entry_dict)
    path = _log_path(repo_name)
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"[log_writer] ERROR: Could not write log '{path}': {exc}", file=sys.stderr)


def get_last_sha(repo_name: str) -> str | None:
    """Return the most recent 'to' SHA from the log, or None if the log is empty."""
    entries = read_log(repo_name)
    if not entries:
        return None
    return entries[-1].get("commit_range", {}).get("to")


def write_file_tree(repo_name: str, tree_dict: dict) -> None:
    """Save the current file tree snapshot for a repo to store/<repo-name>-tree.json."""
    path = _tree_path(repo_name)
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(tree_dict, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"[log_writer] ERROR: Could not write tree '{path}': {exc}", file=sys.stderr)


def write_index(repo_names: list[str]) -> None:
    """Write store/index.json — the manifest of all tracked repo names for the dashboard."""
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    index_path = _STORE_DIR / "index.json"
    try:
        with index_path.open("w", encoding="utf-8") as fh:
            json.dump(repo_names, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"[log_writer] ERROR: Could not write index '{index_path}': {exc}", file=sys.stderr)


def make_entry(
    summary: str,
    commits: list[dict],
    token_usage: dict | None = None,
    next_steps: list[str] | None = None,
) -> dict:
    """Build a log entry dict from a summary string, commits, optional token_usage, and next_steps."""
    all_files: list[str] = []
    total_files = 0
    total_ins = 0
    total_del = 0

    for commit in commits:
        stats = commit.get("stats", {})
        total_files += stats.get("files", 0)
        total_ins += stats.get("insertions", 0)
        total_del += stats.get("deletions", 0)
        for f in commit.get("changed_files", []):
            if f not in all_files:
                all_files.append(f)

    from_sha = commits[0]["sha"] if commits else ""
    to_sha = commits[-1]["sha"] if commits else ""

    entry: dict = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "commit_range": {"from": from_sha, "to": to_sha},
        "summary": summary,
        "next_steps": next_steps or [],
        "files_changed": total_files,
        "insertions": total_ins,
        "deletions": total_del,
        "changed_files": all_files,
        "token_usage": token_usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    return entry


# ── Token usage log ────────────────────────────────────────────────────────────

def _token_log_path() -> Path:
    """Return the Path for the shared token-usage log file."""
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    return _STORE_DIR / "token_usage.json"


def read_token_log() -> list[dict]:
    """Load the full token-usage log, returning an empty list if it does not exist."""
    path = _token_log_path()
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[log_writer] WARNING: Could not read token log '{path}': {exc}", file=sys.stderr)
        return []


def append_token_log(repo_name: str, model: str, token_usage: dict) -> None:
    """Append one token-usage record to store/token_usage.json for later review."""
    records = read_token_log()
    next_id = (records[-1]["id"] + 1) if records else 1
    record = {
        "id": next_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "repo": repo_name,
        "model": model,
        "prompt_tokens": token_usage.get("prompt_tokens", 0),
        "completion_tokens": token_usage.get("completion_tokens", 0),
        "total_tokens": token_usage.get("total_tokens", 0),
    }
    records.append(record)
    path = _token_log_path()
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"[log_writer] ERROR: Could not write token log '{path}': {exc}", file=sys.stderr)


def get_token_summary() -> dict:
    """Return aggregate token totals across all runs, broken down by repo and overall."""
    records = read_token_log()
    overall = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0}
    by_repo: dict[str, dict] = {}

    for rec in records:
        repo = rec.get("repo", "unknown")
        if repo not in by_repo:
            by_repo[repo] = {
                "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "api_calls": 0,
            }
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            by_repo[repo][key] += rec.get(key, 0)
            overall[key] += rec.get(key, 0)
        by_repo[repo]["api_calls"] += 1
        overall["api_calls"] += 1

    return {"overall": overall, "by_repo": by_repo}
