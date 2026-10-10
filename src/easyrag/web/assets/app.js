// EasyRAG chat page. Vanilla JS, no build step (ARCHITECTURE D24, D37).
// All text from the server is inserted with textContent, never as HTML.

import {
  errorMessage,
  lineRange,
  paragraphs,
  parseInline,
  readinessProblems,
  sourceLabel,
  statusNotice,
  traceSummary,
} from "./render.js";

const KEY_STORAGE = "easyrag.apiKey";

const $ = (selector) => document.querySelector(selector);
const conversation = $("#conversation");
const form = $("#ask");
const question = $("#question");
const submit = $("#submit");
const stop = $("#stop");
const banner = $("#banner");
const indexInfo = $("#index-info");
const keyButton = $("#key-button");
const keyDialog = $("#key-dialog");
const keyInput = $("#key-input");
const keyError = $("#key-error");
$("#key-cancel").addEventListener("click", () => keyDialog.close("cancel"));

let entryCount = 0;

// -- API key (this tab only) ---------------------------------------------------------

function storedKey() {
  try {
    return sessionStorage.getItem(KEY_STORAGE);
  } catch {
    return null; // storage unavailable (private mode, blocked): ask again when needed
  }
}

function storeKey(key) {
  try {
    if (key) sessionStorage.setItem(KEY_STORAGE, key);
    else sessionStorage.removeItem(KEY_STORAGE);
  } catch {
    /* ignore: the key then lasts until the next 401 */
  }
  updateKeyButton();
}

let memoryKey = storedKey();

function currentKey() {
  return storedKey() || memoryKey;
}

function updateKeyButton(required = false) {
  const hasKey = Boolean(currentKey());
  keyButton.hidden = !hasKey && !required;
  keyButton.textContent = hasKey ? "Forget API key" : "Enter API key";
}

/** Ask for the key; resolves true when one was entered. */
function askForKey(message) {
  keyError.textContent = message || "";
  keyInput.value = "";
  keyDialog.returnValue = ""; // Escape keeps the previous value; start from "not saved"
  keyDialog.showModal();
  keyInput.focus();
  return new Promise((resolve) => {
    keyDialog.addEventListener(
      "close",
      () => {
        const key = keyDialog.returnValue === "save" ? keyInput.value.trim() : "";
        if (key) {
          memoryKey = key;
          storeKey(key);
          loadStatus(); // the index size needs the key too
        }
        resolve(Boolean(key));
      },
      { once: true },
    );
  });
}

keyButton.addEventListener("click", () => {
  if (currentKey()) {
    memoryKey = null;
    storeKey(null);
    updateKeyButton(true);
  } else {
    askForKey();
  }
});

// -- HTTP --------------------------------------------------------------------------

/** fetch JSON; on 401 asks for the key and retries once. Returns {ok, status, body, requestId}. */
async function api(path, { method = "GET", body, signal, interactive = true } = {}) {
  for (let attempt = 0; attempt < 2; attempt += 1) {
    const headers = { Accept: "application/json" };
    if (body !== undefined) headers["Content-Type"] = "application/json";
    const key = currentKey();
    if (key) headers["X-API-Key"] = key;

    let response;
    try {
      response = await fetch(path, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal,
      });
    } catch (error) {
      if (error.name === "AbortError") throw error;
      return { ok: false, status: 0, body: null, requestId: null };
    }
    const requestId = response.headers.get("X-Request-ID");
    let data = null;
    try {
      data = await response.json();
    } catch {
      /* not JSON */
    }
    if (response.status === 401) {
      updateKeyButton(true);
      if (interactive && attempt === 0) {
        const message = key ? "That API key was not accepted." : "";
        if (await askForKey(message)) continue;
      }
    }
    return { ok: response.ok, status: response.status, body: data, requestId };
  }
  return { ok: false, status: 401, body: null, requestId: null };
}

// -- Status: readiness banner and index size ----------------------------------------

async function loadStatus() {
  const ready = await api("/health/ready", { interactive: false });
  const problems = ready.body ? readinessProblems(ready.body.checks) : [];
  if (ready.status === 0) problems.push("Cannot reach the EasyRAG server.");
  banner.replaceChildren(...problems.map((text) => element("p", {}, text)));
  banner.hidden = problems.length === 0;

  const status = await api("/api/v1/index/status", { interactive: false });
  if (status.ok && status.body) {
    const { documents, chunks, state } = status.body;
    const running = state === "running" ? " · indexing…" : "";
    indexInfo.textContent = `${documents} documents · ${chunks} sections${running}`;
  } else {
    indexInfo.textContent = "";
  }
}

// -- Rendering -----------------------------------------------------------------------

function element(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (name === "class") node.className = value;
    else node.setAttribute(name, value);
  }
  for (const child of children) {
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function renderAnswerText(container, text, sources, entryId) {
  for (const paragraph of paragraphs(text)) {
    const p = element("p");
    for (const segment of parseInline(paragraph, sources.length)) {
      if (segment.type === "text") {
        p.append(document.createTextNode(segment.value));
      } else if (segment.type === "code") {
        p.append(element("code", {}, segment.value));
      } else {
        const button = element(
          "button",
          { type: "button", class: "cite", "aria-label": `Show source ${segment.n}` },
          `[${segment.n}]`,
        );
        button.addEventListener("click", () => openSource(entryId, segment.n));
        p.append(button);
      }
    }
    container.append(p);
  }
}

function renderSources(sources, entryId) {
  const list = element("ol", { class: "sources" });
  sources.forEach((source, index) => {
    const n = index + 1;
    const details = element("details", { id: `${entryId}-source-${n}` });
    details.append(
      element(
        "summary",
        {},
        element("span", { class: "source-number" }, `[${n}]`),
        element("span", { class: "source-label" }, sourceLabel(source)),
        element("span", { class: "source-lines" }, lineRange(source.lines)),
      ),
      element("pre", { class: "excerpt" }, source.content),
    );
    list.append(element("li", {}, details));
  });
  return list;
}

function openSource(entryId, n) {
  const details = document.getElementById(`${entryId}-source-${n}`);
  if (!details) return;
  details.open = true;
  details.scrollIntoView({ behavior: "smooth", block: "nearest" });
  details.classList.remove("flash");
  void details.offsetWidth; // restart the highlight animation
  details.classList.add("flash");
}

function renderAnswer(entry, answer) {
  const entryId = entry.id;
  const body = element("div", { class: "answer" });
  const notice = statusNotice(answer);
  if (notice) body.append(element("p", { class: `notice ${notice.tone}` }, notice.text));
  if (answer.status !== "insufficient_evidence") {
    renderAnswerText(body, answer.answer, answer.sources, entryId);
  }
  if (answer.truncated) {
    body.append(element("p", { class: "notice muted" }, "The answer was cut off at the output limit."));
  }
  if (answer.sources.length) {
    body.append(element("h3", {}, answer.sources.length === 1 ? "Source" : "Sources"));
    body.append(renderSources(answer.sources, entryId));
  }
  const summary = traceSummary(answer.trace);
  if (summary) body.append(element("p", { class: "trace" }, summary));
  entry.querySelector(".pending").replaceWith(body);
}

function renderError(entry, result) {
  const message = errorMessage(result.status, result.body);
  const body = element("div", { class: "answer" }, element("p", { class: "notice error" }, message));
  if (result.requestId) {
    body.append(element("p", { class: "trace" }, `Request ID: ${result.requestId}`));
  }
  entry.querySelector(".pending").replaceWith(body);
}

// -- Asking --------------------------------------------------------------------------

let controller = null;

async function ask(text) {
  entryCount += 1;
  const entry = element("article", { class: "entry", id: `entry-${entryCount}` });
  const seconds = element("span", {}, "0");
  const pending = element("p", { class: "pending" }, "Searching your documents and writing an answer… ", seconds, " s");
  entry.append(element("h2", { class: "question" }, text), pending);
  conversation.append(entry);
  entry.scrollIntoView({ behavior: "smooth", block: "start" });

  const started = Date.now();
  const timer = setInterval(() => {
    seconds.textContent = String(Math.round((Date.now() - started) / 1000));
  }, 1000);
  controller = new AbortController();
  setBusy(true);
  try {
    const result = await api("/api/v1/chat", {
      method: "POST",
      body: { question: text, debug: true },
      signal: controller.signal,
    });
    if (result.ok && result.body) renderAnswer(entry, result.body);
    else renderError(entry, result);
  } catch (error) {
    if (error.name !== "AbortError") throw error;
    entry.querySelector(".pending").replaceWith(element("p", { class: "notice muted" }, "Stopped."));
  } finally {
    clearInterval(timer);
    controller = null;
    setBusy(false);
  }
}

function setBusy(busy) {
  submit.disabled = busy;
  stop.hidden = !busy;
  conversation.setAttribute("aria-busy", String(busy));
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = question.value.trim();
  if (!text || controller) return;
  question.value = "";
  ask(text).finally(() => question.focus());
});

question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

stop.addEventListener("click", () => controller && controller.abort());

updateKeyButton();
loadStatus();
question.focus();
