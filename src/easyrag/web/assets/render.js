// Pure helpers for the chat page: no DOM, so they are unit-tested with Node
// (tests/web/render.test.mjs). Nothing here produces HTML: the page builds DOM nodes from
// these segments with textContent, so model output can never become markup.

/**
 * Split one paragraph of answer text into segments:
 *   {type: "text", value} | {type: "code", value} | {type: "cite", n}
 * `code` is a `backtick` span. `[n]` becomes a citation only when 1 <= n <= sourceCount;
 * anything else stays text. Citations are not recognised inside code spans.
 */
export function parseInline(text, sourceCount) {
  const segments = [];
  const pushText = (value) => {
    if (!value) return;
    const last = segments[segments.length - 1];
    if (last && last.type === "text") last.value += value;
    else segments.push({ type: "text", value });
  };

  const pushProse = (value) => {
    let last = 0;
    for (const match of value.matchAll(/\[(\d{1,3})\]/g)) {
      const n = Number(match[1]);
      if (n >= 1 && n <= sourceCount) {
        pushText(value.slice(last, match.index));
        segments.push({ type: "cite", n });
        last = match.index + match[0].length;
      }
    }
    pushText(value.slice(last));
  };

  const parts = text.split("`");
  // An unmatched trailing backtick is literal text, not the start of a code span.
  const closed = parts.length % 2 === 1 ? parts.length : parts.length - 1;
  parts.forEach((part, index) => {
    if (index >= closed) {
      pushText("`");
      pushProse(part);
    } else if (index % 2 === 1) {
      if (part) segments.push({ type: "code", value: part });
      else pushText("``");
    } else {
      pushProse(part);
    }
  });
  return segments;
}

/** Paragraphs separated by blank lines; single newlines are kept inside a paragraph. */
export function paragraphs(text) {
  return text
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter(Boolean);
}

/** "guides/restore.md › Restore › Partial restores" */
export function sourceLabel(source) {
  const path = source.heading_path && source.heading_path.length ? source.heading_path : [source.section];
  return [source.document, ...path].join(" › ");
}

export function lineRange(lines) {
  const [start, end] = lines;
  return start === end ? `line ${start}` : `lines ${start}–${end}`;
}

/** How to present an answer's status: null for a normal, grounded answer. */
export function statusNotice(answer) {
  if (answer.status === "insufficient_evidence") {
    return { tone: "muted", text: "Not found in your documents." };
  }
  if (answer.status === "ungrounded") {
    return { tone: "warning", text: "Not verified: this answer cites none of your documents." };
  }
  return null;
}

/** A readable message from an API error response ({error: {code, message}}). */
export function errorMessage(status, body) {
  const error = body && typeof body === "object" ? body.error : null;
  if (error && typeof error.message === "string") return error.message;
  if (status === 0) return "Cannot reach the EasyRAG server.";
  return `The server answered with HTTP ${status}.`;
}

/** One line of trace details: "4.2 s · retrieval 0.31 s · 850 + 42 tokens · top score 0.031" */
export function traceSummary(trace) {
  if (!trace) return "";
  const parts = [`${(trace.total_ms / 1000).toFixed(1)} s`];
  parts.push(`retrieval ${(trace.retrieval_ms / 1000).toFixed(2)} s`);
  if (trace.prompt_tokens != null && trace.completion_tokens != null) {
    parts.push(`${trace.prompt_tokens} + ${trace.completion_tokens} tokens`);
  }
  if (trace.retrieved && trace.retrieved.length) {
    parts.push(`top score ${trace.retrieved[0].score.toFixed(3)}${trace.reranked ? " (reranked)" : ""}`);
  }
  if (trace.llm_model) parts.push(trace.llm_model);
  return parts.join(" · ");
}

/** Readiness checks that are not "ok", as user-facing sentences. */
export function readinessProblems(checks) {
  const explain = {
    database: "The database is unavailable.",
    schema: "The database schema is out of date (migrations pending).",
    embedding_model: "The embedding model is not available yet (it may still be downloading).",
    llm_model: "The chat model is not available yet (it may still be downloading).",
    index: "The document index is not ready (indexing may still be running).",
    reranker: "The reranker is still loading; answers work without it meanwhile.",
  };
  const pulling = {
    embedding_model: "The embedding model is downloading (first start); indexing follows.",
    llm_model: "The chat model is downloading (first start); questions can be asked when it is done.",
  };
  return Object.entries(checks || {})
    .filter(([, status]) => status !== "ok")
    .map(([name, status]) =>
      status === "pulling" && pulling[name] ? pulling[name] : explain[name] || `${name}: ${status}`,
    );
}
