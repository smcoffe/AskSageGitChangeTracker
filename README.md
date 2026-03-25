# Git Intel

Git Intel is a local tool that monitors one or more git repositories, uses a language model to turn recent commit history into plain-English summaries, stores those summaries in a running JSON log, and serves a clean browser dashboard showing a live file tree and scrollable change timeline for every tracked repository.

It supports three LLM backends:
- **OpenAI hosted API** — set `openai_api_key` and leave `base_url` empty.
- **Local vLLM server** — set `base_url` to your vLLM endpoint and leave `openai_api_key` empty.
- **Ask Sage API** — set `provider` to `"asksage"` and provide your `asksage_email` and `asksage_api_key`.

---

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure**

   Open `config.json` and fill in your details:

   **Option A — OpenAI hosted API**
   - Set `"provider"` to `"openai"` (or omit it — `"openai"` is the default).
   - Set `"openai_api_key"` to your [OpenAI API key](https://platform.openai.com/api-keys).
   - Leave `"base_url"` as an empty string or omit it.
   - Set `"model"` to the desired OpenAI model (e.g. `"gpt-4o-mini"`).

   **Option B — Local vLLM server**
   - Set `"provider"` to `"openai"` (or omit it).
   - Set `"base_url"` to your vLLM server's OpenAI-compatible endpoint, including the `/v1` path (e.g. `"http://192.168.1.50:8000/v1"`).
   - Leave `"openai_api_key"` as an empty string — vLLM ignores it.
   - Set `"model"` to the Hugging Face model ID that was passed to vLLM when starting the server (e.g. `"mistralai/Mistral-7B-Instruct-v0.2"`).

   **Option C — Ask Sage API**
   - Set `"provider"` to `"asksage"`.
   - Set `"asksage_email"` to the email associated with your Ask Sage account.
   - Set `"asksage_api_key"` to your Ask Sage static API key.
   - Set `"model"` to a model available on Ask Sage (e.g. `"gpt-4o"`, `"claude-3.5-sonnet"`). You can discover available models via the Ask Sage /get-models endpoint.
   - The `openai_api_key` and `base_url` fields are ignored when using Ask Sage.

   Then replace the placeholder entry in `"repositories"` with the actual name and **absolute path** of each local git repository you want to track, and adjust `schedule_minutes`, `lookback_commits`, and other settings as desired.

   > ⚠️ Never commit `config.json` (containing API keys) to version control.

### Starting a vLLM server (quick reference)

If you don't already have a vLLM server running on your network, here's the minimal command to start one:

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --host 0.0.0.0 \
    --port 8000
```

Once running, set `"base_url": "http://<server-ip>:8000/v1"` in `config.json`.

---

## Running

### Start the monitor

```bash
python main.py
```

Git Intel will print a startup summary, immediately process all configured repos, write summaries to `store/`, and — if `schedule_minutes > 0` — continue polling on the configured interval. Press **Ctrl+C** to stop.

### Serve the dashboard

Open a second terminal in the `git-intel/` root directory and run:

```bash
python -m http.server 5500
```

Then open your browser to:

```
http://localhost:5500/dashboard/
```

The dashboard reads from the `store/` directory, which is accessible at the server root, so the two terminal sessions can run simultaneously.

---

## Config reference

| Key | Type | Description |
|---|---|---|
| `provider` | string | LLM provider: `"openai"` (default, also works with vLLM) or `"asksage"`. |
| `openai_api_key` | string | OpenAI API key. Required when provider is `"openai"` and `base_url` is empty. Leave empty when using vLLM or Ask Sage. |
| `base_url` | string | Base URL of an OpenAI-compatible API, including `/v1` (e.g. `"http://192.168.1.50:8000/v1"`). Leave empty to use the hosted OpenAI API. Only used when provider is `"openai"`. |
| `asksage_email` | string | Ask Sage account email. Required when provider is `"asksage"`. |
| `asksage_api_key` | string | Ask Sage static API key. Required when provider is `"asksage"`. |
| `model` | string | Model name. For OpenAI: `gpt-4o-mini`, `gpt-4o`, etc. For vLLM: the Hugging Face model ID. For Ask Sage: any model from the /get-models endpoint. |
| `repositories` | array | List of repos to monitor. Each entry needs `"name"` (display label) and `"path"` (absolute path on disk). |
| `schedule_minutes` | integer | How often to check for new commits, in minutes. Set to `0` to run once and exit. |
| `lookback_commits` | integer | Max number of recent commits to process on the very first run per repo. |
| `dashboard_port` | integer | Informational — the port shown in the startup message. Pass this to `http.server` manually. |

---

## Dashboard

### Opening it

Serve from the `git-intel/` root with `python -m http.server 5500`, then visit `http://localhost:5500/dashboard/`.

### Panels

| Panel | Description |
|---|---|
| **Repository selector** (top of sidebar) | Switch between tracked repositories. |
| **File Tree** (sidebar) | The current working tree of the selected repo at the time of the last `main.py` run. Folders are shown with `▸`, files with `·`. |
| **Change Timeline** (main area) | Plain-English summaries of commit batches in reverse-chronological order. Each entry shows the date/time, a prose summary, and stat badges for files changed, insertions, and deletions. |
| **↻ Sync button** | Re-fetches the latest data from `store/` without reloading the page. |
| **Last updated** (footer) | Timestamp of the most recent dashboard refresh. |

---

## Security note

If using the OpenAI hosted API, your API key is stored in `config.json` in plain text. The same applies to Ask Sage credentials (`asksage_email` and `asksage_api_key`). Add `config.json` to your `.gitignore` before committing this project anywhere, or use a separate secrets manager. When using a local vLLM server no API key is required and this concern does not apply.
