/**
 * Git Intel — dashboard client-side logic.
 * Vanilla JS, no external dependencies.
 */

/* ── Helpers ── */

/**
 * Formats an ISO 8601 date string into a human-readable local date/time.
 * @param {string} iso
 * @returns {string}
 */
function formatDate(iso) {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

/**
 * Escape a string for safe insertion as HTML text content.
 * @param {string} str
 * @returns {string}
 */
function esc(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* ── Rendering ── */

/**
 * Render timeline entries in reverse-chronological order into #timeline.
 * @param {Array<Object>} entries
 */
function renderTimeline(entries) {
  const container = document.getElementById("timeline");
  container.innerHTML = "";

  if (!entries || entries.length === 0) {
    container.innerHTML =
      '<p id="empty-state">No data yet. Run <code>main.py</code> to generate summaries.</p>';
    return;
  }

  // Reverse so newest is first
  const sorted = [...entries].sort((a, b) => (b.id || 0) - (a.id || 0));

  sorted.forEach((entry) => {
    const div = document.createElement("div");
    div.className = "entry";

    const files = entry.files_changed ?? 0;
    const ins   = entry.insertions   ?? 0;
    const del   = entry.deletions    ?? 0;

    // Build the next-steps section if the entry has suggestions
    const steps = Array.isArray(entry.next_steps) ? entry.next_steps : [];
    let stepsHtml = "";
    if (steps.length > 0) {
      const items = steps.map((s) => `<li>${esc(s)}</li>`).join("");
      stepsHtml = `
        <div class="entry-next-steps">
          <div class="entry-steps-header">💡 Suggested Next Steps</div>
          <ol class="steps-list">${items}</ol>
        </div>`;
    }

    div.innerHTML = `
      <div class="entry-date">${esc(formatDate(entry.timestamp))}</div>
      <p class="entry-summary">${esc(entry.summary ?? "")}</p>
      <div class="entry-badges">
        <span class="badge badge-files">📄 ${esc(String(files))} file${files !== 1 ? "s" : ""} changed</span>
        <span class="badge badge-ins">+${esc(String(ins))} insertion${ins !== 1 ? "s" : ""}</span>
        <span class="badge badge-del">−${esc(String(del))} deletion${del !== 1 ? "s" : ""}</span>
      </div>
      ${stepsHtml}
    `;
    container.appendChild(div);
  });
}

/**
 * Recursively render a file-tree node into a <ul> element.
 * @param {Object} node  - { name, type, children }
 * @param {number} depth - current recursion depth (0 = root)
 * @returns {HTMLUListElement|null}
 */
function renderTree(node, depth) {
  if (!node) return null;

  const ul = document.createElement("ul");

  // Skip the invisible root wrapper — render its children directly
  const items = (depth === 0 && node.name === "") ? (node.children || []) : [node];

  items.forEach((item) => {
    const li = document.createElement("li");

    if (item.type === "dir") {
      const icon = document.createElement("span");
      icon.className = "tree-icon";
      icon.textContent = "▸";
      const label = document.createElement("span");
      label.className = "tree-dir";
      label.textContent = item.name;
      li.appendChild(icon);
      li.appendChild(label);

      if (item.children && item.children.length > 0) {
        const childUl = document.createElement("ul");
        item.children.forEach((child) => {
          const childRendered = renderTree(child, depth + 1);
          if (childRendered) {
            // childRendered is a <ul> with one <li>; unwrap it
            Array.from(childRendered.children).forEach((c) => childUl.appendChild(c));
          }
        });
        li.appendChild(childUl);
      }
    } else {
      const icon = document.createElement("span");
      icon.className = "tree-icon";
      icon.textContent = "·";
      const label = document.createElement("span");
      label.className = "tree-file";
      label.textContent = item.name;
      li.appendChild(icon);
      li.appendChild(label);
    }

    ul.appendChild(li);
  });

  return ul;
}

/* ── Data loading ── */

/**
 * Fetch JSON from a path, returning null on error instead of throwing.
 * @param {string} path
 * @returns {Promise<any|null>}
 */
async function fetchJSON(path) {
  try {
    const resp = await fetch(path);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();
  } catch (err) {
    console.warn(`[git-intel] Could not fetch ${path}:`, err);
    return null;
  }
}

/**
 * Load data for the selected repo: timeline entries and file tree.
 * @param {string} repoName
 */
async function loadRepo(repoName) {
  if (!repoName) return;

  const slug = repoName.toLowerCase().replace(/[^\w\s-]/g, "").replace(/[\s_]+/g, "-");

  const [entries, tree] = await Promise.all([
    fetchJSON(`/store/${slug}.json`),
    fetchJSON(`/store/${slug}-tree.json`),
  ]);

  // Timeline
  if (entries) {
    renderTimeline(entries);
  } else {
    const container = document.getElementById("timeline");
    container.innerHTML =
      '<p id="empty-state">No data yet. Run <code>main.py</code> to generate summaries.</p>';
  }

  // File tree
  const treeContainer = document.getElementById("file-tree");
  treeContainer.innerHTML = "";
  if (tree) {
    const rendered = renderTree(tree, 0);
    if (rendered) treeContainer.appendChild(rendered);
  } else {
    treeContainer.innerHTML = '<p style="padding:8px;color:var(--text-muted);font-size:13px;">No file tree available.</p>';
  }

  // Last-updated stamp
  document.getElementById("last-updated").textContent = new Date().toLocaleTimeString();
}

/**
 * Load the repo manifest from /store/index.json and populate the selector.
 */
async function loadRepos() {
  const repos = await fetchJSON("/store/index.json");

  const select = document.getElementById("repo-select");
  select.innerHTML = "";

  if (!repos || repos.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No repos found — run main.py";
    select.appendChild(opt);

    document.getElementById("timeline").innerHTML =
      '<p id="empty-state">No data yet. Run <code>main.py</code> to generate summaries.</p>';
    return;
  }

  repos.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    select.appendChild(opt);
  });

  // Load the first repo immediately
  await loadRepo(repos[0]);
}

/* ── Event wiring ── */

document.addEventListener("DOMContentLoaded", () => {
  loadRepos();

  document.getElementById("repo-select").addEventListener("change", (e) => {
    loadRepo(e.target.value);
  });

  document.getElementById("sync-btn").addEventListener("click", () => {
    const selected = document.getElementById("repo-select").value;
    loadRepo(selected);
  });
});
