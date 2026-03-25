# NOTES — Design Decisions, Deviations & Limitations

## Design decisions where the spec was ambiguous

### Commit ordering
The spec says "return only commits newer than that SHA (exclusive)" but doesn't specify the ordering of the returned list. I chose **oldest-first** (chronological) throughout `git_parser.get_commits()`. This makes the `commit_range.from` / `to` fields in each log entry semantically correct (from = oldest SHA, to = newest SHA), and it means `get_last_sha()` can simply read the `to` field of the last entry to correctly resume on the next run.

### Batch granularity — one summary per run
The spec says to "summarise a batch of commits". I treat each invocation of `run_once()` per repo as one batch: all new commits since the last run are summarised in a single LLM call. This is cost-efficient and produces a coherent paragraph. An alternative would be to summarise each commit individually or in fixed-size windows; that is easy to layer on top by changing the loop in `main.py`.

### `git_parser.get_file_tree` — root node name
When the root `Path` is the repo directory itself, `rel_path` starts as `Path("")` and `rel_path.name` is an empty string. I preserved this as `""` in the root node so `app.js` can detect it and skip directly to rendering the children, avoiding a blank top-level label. The root node is therefore `{"name": "", "type": "dir", "children": [...]}`.

### `log_writer._slugify` — repo name sanitisation
The spec says "slugify the name: lowercase, spaces→hyphens". I also strip characters that are unsafe in filenames (anything not alphanumeric, hyphen, or whitespace) and collapse repeated hyphens/underscores. This keeps filenames clean on all major operating systems without requiring an external library.

### `app.js` slug derivation
The dashboard must derive the same slug as `log_writer._slugify()` from a repo name so it can fetch the right file. I replicated the core logic in JavaScript: lowercase, strip non-word characters, convert spaces/underscores to hyphens. Complex edge cases (e.g. CJK characters) may diverge between the Python and JS implementations; for most repo names the behaviour is identical.

### `store/` directory is inside `git-intel/`
I placed `store/` as a sibling of `main.py` (resolved relative to `__file__`). This means the directory is always found correctly regardless of the current working directory when `python main.py` is invoked, which is a common source of bugs.

### HTTP server root vs. dashboard root
`python -m http.server 5500` must be run from the `git-intel/` root (not from `dashboard/`). This lets the browser fetch both `/dashboard/index.html` and `/store/*.json` from the same origin, avoiding CORS issues entirely. This is documented in the README.

### LLM temperature
I used `temperature=0.4` for the summarisation call. This is low enough to produce consistent, factual summaries while allowing minor variation. The spec did not prescribe a value.

### `max_tokens=400`
The spec requests 3–5 sentence summaries. 400 tokens is generous for that length in English while guarding against unexpectedly large outputs that would inflate costs. This is easily adjustable in `llm_client.py`.

### No `asyncio`
The spec doesn't require concurrent repo processing. I used a straightforward synchronous loop in `run_once()`. For a large number of repos, switching to `asyncio` + `openai`'s async client would speed things up; that is a natural next step.

---

## Deviations from the instructions

- **`_build_tree_node` vs. `os.walk`**: The spec suggests using `os.walk`. I used recursive `Path.iterdir()` instead because it produces a naturally nested structure (matching the required output format) without requiring post-processing to build the tree. The behaviour is equivalent.

- **`log_writer.make_entry` helper**: The spec describes `append_entry` accepting a pre-built dict. I added `make_entry()` as a convenience function called from `main.py` to avoid scattering the entry-assembly logic across files. This is an additive change; `append_entry` still accepts any dict.

---

## Known limitations

1. **Single active branch only** — `git_parser.get_commits()` iterates the repo's `active_branch`. Repos in detached HEAD state will raise an exception (caught and logged). A future version could fall back to `HEAD` directly.

2. **No incremental tree diff** — The file tree is re-snapshotted on every run. For very large repositories this could be slow. A diff-based approach (only recording changed paths) would be more efficient.

3. **API key in plain text** — `config.json` stores the OpenAI key in the clear. Users should add `config.json` to `.gitignore`. A proper secret-store integration (e.g. `keyring`, environment variables) is a recommended next step.

4. **No authentication on the dashboard** — The HTTP server exposes `store/` to anyone who can reach the port. For sensitive codebases, restrict access at the network level or add HTTP basic auth via a reverse proxy.

5. **Rate limiting** — If many repos with many new commits are processed simultaneously, OpenAI rate limits may be hit. The current code does not implement retries or exponential backoff beyond the basic error catch.

6. **Windows path compatibility** — `pathlib.Path` is used throughout for cross-platform safety, but the slugify function and `config.json` path separators have only been tested on Unix-like systems.

---

## Suggested next steps

- Add environment variable override for `openai_api_key` (`OPENAI_API_KEY`) so the key never needs to be in a file.
- Support multiple branches per repo (configurable per-repo in `config.json`).
- Add a search/filter bar to the dashboard timeline.
- Persist `store/` snapshots between runs with a git-like SHA-based deduplication strategy.
- Package as a proper CLI tool with `pyproject.toml` / `setup.cfg` for `pip install git-intel`.
