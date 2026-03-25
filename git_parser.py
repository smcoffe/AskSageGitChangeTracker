"""Parse local git repositories and extract commit data and file trees."""

import os
import sys
from datetime import timezone
from pathlib import Path
from typing import Optional

import git


def get_commits(
    repo_path: str,
    last_seen_sha: Optional[str],
    lookback_commits: int,
) -> list[dict]:
    """Return a list of unprocessed commit dicts from the given repo path."""
    try:
        repo = git.Repo(repo_path, search_parent_directories=False)
    except (git.InvalidGitRepositoryError, git.NoSuchPathError) as exc:
        print(f"[git_parser] ERROR: Cannot open repo at '{repo_path}': {exc}", file=sys.stderr)
        return []

    commits_out: list[dict] = []

    try:
        branch = repo.active_branch
        commit_iter = repo.iter_commits(branch)
    except Exception as exc:
        print(f"[git_parser] ERROR: Cannot iterate commits in '{repo_path}': {exc}", file=sys.stderr)
        return []

    if last_seen_sha is None:
        # First run: take up to lookback_commits most recent commits
        raw_commits = []
        for commit in commit_iter:
            raw_commits.append(commit)
            if len(raw_commits) >= lookback_commits:
                break
        # Return in chronological order (oldest first) so caller can build range correctly
        raw_commits.reverse()
    else:
        # Incremental run: collect commits newer than last_seen_sha (exclusive)
        raw_commits = []
        for commit in commit_iter:
            if commit.hexsha == last_seen_sha:
                break
            raw_commits.append(commit)
        raw_commits.reverse()  # oldest first

    for commit in raw_commits:
        try:
            stats = commit.stats.total
            changed_files = list(commit.stats.files.keys())
        except Exception:
            stats = {"files": 0, "insertions": 0, "deletions": 0}
            changed_files = []

        # Normalise timestamp to ISO 8601 in UTC
        committed_dt = commit.committed_datetime
        if committed_dt.tzinfo is None:
            committed_dt = committed_dt.replace(tzinfo=timezone.utc)
        ts_iso = committed_dt.astimezone(timezone.utc).isoformat()

        commits_out.append(
            {
                "sha": commit.hexsha,
                "author": str(commit.author),
                "timestamp": ts_iso,
                "message": commit.message.strip(),
                "stats": {
                    "files": stats.get("files", 0),
                    "insertions": stats.get("insertions", 0),
                    "deletions": stats.get("deletions", 0),
                },
                "changed_files": changed_files,
            }
        )

    return commits_out


def _build_tree_node(root: Path, rel_path: Path) -> dict:
    """Recursively build a nested dict representing a directory node."""
    abs_path = root / rel_path
    name = rel_path.name if rel_path.name else str(root)

    if abs_path.is_dir():
        children = []
        try:
            entries = sorted(abs_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            entries = []
        for entry in entries:
            if entry.name == ".git":
                continue
            child_rel = rel_path / entry.name
            children.append(_build_tree_node(root, child_rel))
        return {"name": name, "type": "dir", "children": children}
    else:
        return {"name": name, "type": "file", "children": []}


def get_file_tree(repo_path: str) -> dict:
    """Return the current working tree of a repo as a nested dict for JSON serialisation."""
    root = Path(repo_path)
    if not root.is_dir():
        print(f"[git_parser] ERROR: Path does not exist: '{repo_path}'", file=sys.stderr)
        return {"name": root.name, "type": "dir", "children": []}

    return _build_tree_node(root, Path(""))
