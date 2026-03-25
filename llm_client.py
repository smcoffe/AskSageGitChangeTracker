"""Call an LLM API to produce plain-English summaries of git commit batches
and suggest next development steps based on recent changes.

Supports multiple LLM providers:

1. **OpenAI / vLLM** (default) — Uses the OpenAI Python SDK.  Works with both
   the hosted OpenAI service and locally-hosted models served via vLLM (or any
   other OpenAI-compatible server).  Point ``base_url`` at your vLLM instance,
   e.g. ``http://192.168.1.50:8000/v1``, and the same code path is used for both.

2. **Ask Sage** — Uses the Ask Sage REST API (https://api.asksage.ai).  Set
   ``provider="asksage"`` and supply your Ask Sage email + API key in config.json.
"""

import json
import sys
from typing import Optional

import openai

import asksage_client

# Sentinel token_usage dict returned when no API call is made or the call fails.
_ZERO_USAGE: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def build_prompt(commits: list[dict]) -> str:
    """Format a list of commit dicts into a clean prompt string for the LLM."""
    lines = [
        "You are a software-change summariser. Below is a list of git commits.",
        "Write a plain-English paragraph (3–5 sentences) that describes what changed across these commits.",
        "Avoid technical jargon where possible. Mention which files or areas of the codebase were affected.",
        "Do NOT reproduce raw commit SHAs or raw diff output.",
        "",
        "Commits:",
    ]
    for i, commit in enumerate(commits, start=1):
        stats = commit.get("stats", {})
        changed = ", ".join(commit.get("changed_files", [])[:10])
        if len(commit.get("changed_files", [])) > 10:
            changed += f" … and {len(commit['changed_files']) - 10} more"
        lines.append(
            f"{i}. [{commit.get('timestamp', '')}] {commit.get('author', 'unknown')}: "
            f"{commit.get('message', '').splitlines()[0]}"
        )
        lines.append(
            f"   Files changed: {stats.get('files', 0)}, "
            f"+{stats.get('insertions', 0)} / -{stats.get('deletions', 0)}"
        )
        if changed:
            lines.append(f"   Affected files: {changed}")
    lines.append("")
    lines.append("Summary:")
    return "\n".join(lines)


def _extract_usage(response) -> dict:
    """Extract token counts from an OpenAI response object into a plain dict."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return dict(_ZERO_USAGE)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


# ── Ask Sage helpers ───────────────────────────────────────────────────────────

def _get_asksage_token(config: dict) -> str:
    """Obtain an Ask Sage access token from the config credentials."""
    email = str(config.get("asksage_email", ""))
    api_key = str(config.get("asksage_api_key", ""))
    print(f"[llm_client] Email: {email}")
    print(f"[llm_client] API Key: {api_key}")
    if not email or not api_key:
        raise RuntimeError(
            "Ask Sage provider requires 'asksage_email' and 'asksage_api_key' in config.json"
        )
    return asksage_client.get_access_token(email, api_key)


def _summarise_asksage(
    config: dict,
    model: str,
    commits: list[dict],
) -> tuple[str, dict]:
    """Produce a commit summary using the Ask Sage API."""
    token = _get_asksage_token(config)
    prompt = build_prompt(commits)

    system_prompt = (
        "You are a concise technical writer who explains software changes "
        "to a non-technical audience. Use clear, plain language."
    )

    try:
        response_data = asksage_client.query(
            token,
            prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=0.4,
        )
        summary = asksage_client.extract_response_text(response_data)
        if not summary:
            summary = "Summary unavailable — empty response from Ask Sage."

        # Ask Sage does not return granular token counts, so we estimate
        # based on rough word counts (1 token ≈ 0.75 words).
        token_usage = _estimate_token_usage(prompt, summary)
        return summary, token_usage

    except RuntimeError as exc:
        print(f"[llm_client] ERROR: Ask Sage summarise failed: {exc}", file=sys.stderr)
        return "Summary unavailable — API error.", dict(_ZERO_USAGE)
    except Exception as exc:
        print(f"[llm_client] ERROR: Unexpected error during Ask Sage summarisation: {exc}", file=sys.stderr)
        return "Summary unavailable — API error.", dict(_ZERO_USAGE)


def _suggest_next_steps_asksage(
    config: dict,
    model: str,
    summary: str,
    commits: list[dict],
) -> tuple[list[str], dict]:
    """Generate next-step suggestions using the Ask Sage API."""
    token = _get_asksage_token(config)
    prompt = build_next_steps_prompt(summary, commits)

    system_prompt = (
        "You are a senior software engineer who reviews code changes and "
        "suggests practical, actionable next steps to improve a project. "
        "Always respond with a valid JSON array of strings only — no other text."
    )

    try:
        response_data = asksage_client.query(
            token,
            prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=0.5,
        )
        raw = asksage_client.extract_response_text(response_data)
        token_usage = _estimate_token_usage(prompt, raw)

        if not raw:
            return [], token_usage

        # Parse the JSON array the model should have returned
        try:
            steps = json.loads(raw)
            if not isinstance(steps, list):
                steps = [str(steps)]
        except json.JSONDecodeError:
            # Fallback: split on newlines and strip common bullet markers
            steps = [
                line.lstrip("•-*0123456789.) ").strip()
                for line in raw.splitlines()
                if line.strip() and line.strip() not in ("[", "]")
            ]

        steps = [s for s in steps if s]
        return steps, token_usage

    except RuntimeError as exc:
        print(f"[llm_client] ERROR: Ask Sage next-steps failed: {exc}", file=sys.stderr)
        return [], dict(_ZERO_USAGE)
    except Exception as exc:
        print(f"[llm_client] ERROR: Unexpected error during Ask Sage next-steps: {exc}", file=sys.stderr)
        return [], dict(_ZERO_USAGE)


def _estimate_token_usage(prompt: str, completion: str) -> dict:
    """Rough token-count estimate when the API doesn't return usage stats.

    Uses the heuristic that 1 token ≈ 4 characters (common for English text).
    """
    prompt_tokens = max(1, len(prompt) // 4)
    completion_tokens = max(1, len(completion) // 4) if completion else 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


# ── OpenAI / vLLM implementation ──────────────────────────────────────────────

def _summarise_openai(
    api_key: str,
    model: str,
    commits: list[dict],
    base_url: Optional[str] = None,
) -> tuple[str, dict]:
    """Produce a commit summary using an OpenAI-compatible API."""
    prompt = build_prompt(commits)

    try:
        client_kwargs: dict = {"api_key": api_key or "not-needed"}
        if base_url:
            client_kwargs["base_url"] = base_url

        client = openai.OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise technical writer who explains software changes "
                        "to a non-technical audience. Use clear, plain language."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=400,
        )
        summary = response.choices[0].message.content.strip()
        token_usage = _extract_usage(response)
        return summary, token_usage
    except openai.OpenAIError as exc:
        print(f"[llm_client] ERROR: API call failed: {exc}", file=sys.stderr)
        return "Summary unavailable — API error.", dict(_ZERO_USAGE)
    except Exception as exc:
        print(f"[llm_client] ERROR: Unexpected error during summarisation: {exc}", file=sys.stderr)
        return "Summary unavailable — API error.", dict(_ZERO_USAGE)


def _suggest_next_steps_openai(
    api_key: str,
    model: str,
    summary: str,
    commits: list[dict],
    base_url: Optional[str] = None,
) -> tuple[list[str], dict]:
    """Generate next-step suggestions using an OpenAI-compatible API."""
    prompt = build_next_steps_prompt(summary, commits)

    try:
        client_kwargs: dict = {"api_key": api_key or "not-needed"}
        if base_url:
            client_kwargs["base_url"] = base_url

        client = openai.OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior software engineer who reviews code changes and "
                        "suggests practical, actionable next steps to improve a project. "
                        "Always respond with a valid JSON array of strings only — no other text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        token_usage = _extract_usage(response)

        try:
            steps = json.loads(raw)
            if not isinstance(steps, list):
                steps = [str(steps)]
        except json.JSONDecodeError:
            steps = [
                line.lstrip("•-*0123456789.) ").strip()
                for line in raw.splitlines()
                if line.strip() and line.strip() not in ("[", "]")
            ]

        steps = [s for s in steps if s]
        return steps, token_usage

    except openai.OpenAIError as exc:
        print(f"[llm_client] ERROR: Next-steps API call failed: {exc}", file=sys.stderr)
        return [], dict(_ZERO_USAGE)
    except Exception as exc:
        print(f"[llm_client] ERROR: Unexpected error during next-steps generation: {exc}", file=sys.stderr)
        return [], dict(_ZERO_USAGE)


# ── Public API (provider-agnostic) ─────────────────────────────────────────────

def summarise(
    api_key: str,
    model: str,
    commits: list[dict],
    base_url: Optional[str] = None,
    *,
    provider: str = "openai",
    config: Optional[dict] = None,
) -> tuple[str, dict]:
    """Call the configured LLM provider and return (summary_string, token_usage_dict).

    Args:
        api_key:   API key (used for OpenAI/vLLM provider).
        model:     Model name as understood by the selected provider.
        commits:   List of commit dicts produced by git_parser.
        base_url:  Base URL for OpenAI-compatible endpoint (OpenAI/vLLM only).
        provider:  ``"openai"`` (default) or ``"asksage"``.
        config:    Full config dict (needed for Ask Sage credentials).

    Returns:
        A tuple of (summary_string, token_usage_dict).
    """
    if not commits:
        return "No commits to summarise.", dict(_ZERO_USAGE)

    if provider == "asksage":
        if config is None:
            return "Summary unavailable — Ask Sage config not provided.", dict(_ZERO_USAGE)
        return _summarise_asksage(config, model, commits)
    else:
        return _summarise_openai(api_key, model, commits, base_url)


def build_next_steps_prompt(summary: str, commits: list[dict]) -> str:
    """Format a summary and commit list into a prompt asking for actionable next steps."""
    lines = [
        "You are a senior software engineer reviewing recent changes to a codebase.",
        "Based on the recent commit activity described below, suggest 3–5 concrete next steps",
        "that would meaningfully improve this project.",
        "Each step should be a single, actionable sentence a developer could act on immediately.",
        "",
        'Respond with ONLY a JSON array of strings, for example:',
        '["Add unit tests for the new authentication module", "Refactor the database layer to reduce duplication"]',
        "Do NOT include any explanation, preamble, or text outside the JSON array.",
        "",
        f"Recent changes summary: {summary}",
        "",
        "Commits:",
    ]
    for i, commit in enumerate(commits, start=1):
        lines.append(
            f"{i}. [{commit.get('timestamp', '')}] {commit.get('author', 'unknown')}: "
            f"{commit.get('message', '').splitlines()[0]}"
        )
    lines.append("")
    lines.append("Next steps (JSON array only):")
    return "\n".join(lines)


def suggest_next_steps(
    api_key: str,
    model: str,
    summary: str,
    commits: list[dict],
    base_url: Optional[str] = None,
    *,
    provider: str = "openai",
    config: Optional[dict] = None,
) -> tuple[list[str], dict]:
    """Call the configured LLM provider and return (next_steps_list, token_usage_dict).

    Args:
        api_key:   API key (used for OpenAI/vLLM provider).
        model:     Model name as understood by the selected provider.
        summary:   Plain-English summary already generated by ``summarise()``.
        commits:   List of commit dicts produced by git_parser.
        base_url:  Base URL for OpenAI-compatible endpoint (OpenAI/vLLM only).
        provider:  ``"openai"`` (default) or ``"asksage"``.
        config:    Full config dict (needed for Ask Sage credentials).

    Returns:
        A tuple of (next_steps_list, token_usage_dict).
    """
    if not commits:
        return [], dict(_ZERO_USAGE)

    if provider == "asksage":
        if config is None:
            return [], dict(_ZERO_USAGE)
        return _suggest_next_steps_asksage(config, model, summary, commits)
    else:
        return _suggest_next_steps_openai(api_key, model, summary, commits, base_url)
