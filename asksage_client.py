"""Ask Sage API client for authenticating and querying the Ask Sage LLM service.

Ask Sage (https://asksage.ai) is a generative AI platform that provides access to
various LLM models through a REST API.  Authentication uses either a static API key
or a 24-hour access token obtained via the /get-token-with-api-key endpoint.

Server API base URL: https://api.asksage.ai/server/
"""

import sys
import time
from typing import Optional

import requests

# ── Constants ──────────────────────────────────────────────────────────────────

_SERVER_BASE = "https://api.asksage.ai/server"
_USER_BASE = "https://api.asksage.ai/user"
_TOKEN_ENDPOINT = f"{_USER_BASE}/get-token-with-api-key"
_QUERY_ENDPOINT = f"{_SERVER_BASE}/query"
_MODELS_ENDPOINT = f"{_SERVER_BASE}/get-models"

# Cache the access token so we don't re-auth on every call within the same run.
_cached_token: Optional[str] = None
_token_expiry: float = 0.0  # epoch seconds


# ── Authentication ─────────────────────────────────────────────────────────────

def get_access_token(email: str, api_key: str, *, force_refresh: bool = False) -> str:
    """Obtain (or return cached) 24-hour access token from Ask Sage.

    Args:
        email:    The email address associated with the Ask Sage account.
        api_key:  The static API key for the account.
        force_refresh:  If True, ignore the cache and request a new token.

    Returns:
        A string access token suitable for the ``x-access-tokens`` header.

    Raises:
        RuntimeError: If the token request fails.
    """
    global _cached_token, _token_expiry

    # Return cached token if it's still fresh (with 5-minute safety margin)
    if not force_refresh and _cached_token and time.time() < _token_expiry - 300:
        return _cached_token

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "email": email,
        "api_key": api_key,
    }

    try:
        resp = requests.post(_TOKEN_ENDPOINT, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Ask Sage token request failed: {exc}") from exc

    token = data.get("response") or data.get("token") or data.get("access_token")
    if not token:
        raise RuntimeError(
            f"Ask Sage token response did not contain a token. Response: {data}"
        )

    _cached_token = token
    # Tokens are valid for ~24 hours; cache for 23 hours to be safe.
    _token_expiry = time.time() + 23 * 3600
    return token


# ── Query ──────────────────────────────────────────────────────────────────────

def query(
    token: str,
    message: str,
    *,
    model: str = "gpt-4o",
    system_prompt: str = "",
    temperature: float = 0.4,
    dataset: Optional[list[str]] = None,
) -> dict:
    """Send a query to the Ask Sage /query endpoint and return the full response dict.

    Args:
        token:          Access token (from ``get_access_token``).
        message:        The user message / prompt to send.
        model:          Model name as returned by /get-models (e.g. ``"gpt-4o"``).
        system_prompt:  Optional system-level instruction prepended to the message.
        temperature:    Sampling temperature (0.0 – 1.0).
        dataset:        Optional list of dataset names for RAG context.

    Returns:
        The full JSON response dict from Ask Sage.  The actual LLM output is
        typically in ``response["response"]``.

    Raises:
        RuntimeError: If the API call fails or returns an error.
    """
    headers = {
        "x-access-tokens": token,
        "Content-Type": "application/json",
    }

    # Build the combined message: system prompt + user message
    if system_prompt:
        full_message = f"[System]: {system_prompt}\n\n[User]: {message}"
    else:
        full_message = message

    payload: dict = {
        "message": full_message,
        "model": model,
        "temperature": temperature,
        "dataset": dataset or [],
        "persona": 0,           # 0 = no persona
        "limit_references": 0,  # No RAG references needed for git summaries
        "live": 0,              # No live web search needed
    }

    try:
        resp = requests.post(_QUERY_ENDPOINT, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Ask Sage query failed: {exc}") from exc

    # Check for API-level errors in the response body
    if data.get("status") == "error" or data.get("error"):
        error_msg = data.get("error") or data.get("message") or str(data)
        raise RuntimeError(f"Ask Sage API error: {error_msg}")

    return data


def extract_response_text(response_data: dict) -> str:
    """Extract the LLM-generated text from an Ask Sage query response.

    The response structure may vary slightly; this tries several known keys.
    """
    # The primary response field
    text = response_data.get("response")
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Fallback: check for 'message' key
    text = response_data.get("message")
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Fallback: check nested 'data' → 'response'
    data = response_data.get("data", {})
    if isinstance(data, dict):
        text = data.get("response")
        if isinstance(text, str) and text.strip():
            return text.strip()

    return ""


# ── Model listing (utility) ───────────────────────────────────────────────────

def get_available_models(token: str) -> list[str]:
    """Fetch the list of available model names from Ask Sage.

    Returns:
        A list of model name strings, or an empty list on failure.
    """
    headers = {
        "x-access-tokens": token,
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(_MODELS_ENDPOINT, json={}, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("response") or data.get("models") or []
        if isinstance(models, list):
            return models
        return []
    except requests.RequestException as exc:
        print(f"[asksage_client] WARNING: Could not fetch models: {exc}", file=sys.stderr)
        return []
