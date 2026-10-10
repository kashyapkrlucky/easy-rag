// Unit tests for the chat page's pure helpers. Run: node --test tests/web/
// (also run by tests/unit/test_web_ui.py when Node is installed).
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  errorMessage,
  lineRange,
  paragraphs,
  parseInline,
  readinessProblems,
  sourceLabel,
  statusNotice,
  traceSummary,
} from "../../src/easyrag/web/assets/render.js";

test("valid citation markers become citations", () => {
  assert.deepEqual(parseInline("Use compose [1]. Also [2].", 2), [
    { type: "text", value: "Use compose " },
    { type: "cite", n: 1 },
    { type: "text", value: ". Also " },
    { type: "cite", n: 2 },
    { type: "text", value: "." },
  ]);
});

test("out-of-range and zero markers stay text", () => {
  assert.deepEqual(parseInline("See [0] and [3] and [12].", 2), [
    { type: "text", value: "See [0] and [3] and [12]." },
  ]);
});

test("adjacent citations", () => {
  assert.deepEqual(parseInline("x [1][2]", 2), [
    { type: "text", value: "x " },
    { type: "cite", n: 1 },
    { type: "cite", n: 2 },
  ]);
});

test("code spans are kept and not scanned for citations", () => {
  assert.deepEqual(parseInline("Run `docker compose up [1]` now [1].", 1), [
    { type: "text", value: "Run " },
    { type: "code", value: "docker compose up [1]" },
    { type: "text", value: " now " },
    { type: "cite", n: 1 },
    { type: "text", value: "." },
  ]);
});

test("an unmatched backtick is literal text, and citations after it still work", () => {
  assert.deepEqual(parseInline("Use `a` or `b [1]", 1), [
    { type: "text", value: "Use " },
    { type: "code", value: "a" },
    { type: "text", value: " or `b " },
    { type: "cite", n: 1 },
  ]);
});

test("markup in model output is plain text, never HTML", () => {
  const segments = parseInline('<img src=x onerror="alert(1)"> [1]', 1);
  assert.equal(segments[0].type, "text");
  assert.equal(segments[0].value, '<img src=x onerror="alert(1)"> ');
});

test("paragraphs split on blank lines", () => {
  assert.deepEqual(paragraphs("a\nb\n\n  \nc\n"), ["a\nb", "c"]);
});

test("source label and line range", () => {
  const source = { document: "guides/restore.md", heading_path: ["Restore", "Partial"], section: "Partial" };
  assert.equal(sourceLabel(source), "guides/restore.md › Restore › Partial");
  assert.equal(sourceLabel({ ...source, heading_path: [] }), "guides/restore.md › Partial");
  assert.equal(lineRange([4, 4]), "line 4");
  assert.equal(lineRange([4, 9]), "lines 4–9");
});

test("status notices", () => {
  assert.equal(statusNotice({ status: "answered" }), null);
  assert.equal(statusNotice({ status: "insufficient_evidence" }).tone, "muted");
  assert.equal(statusNotice({ status: "ungrounded" }).tone, "warning");
});

test("error messages come from the error envelope", () => {
  assert.equal(errorMessage(503, { error: { code: "llm_unavailable", message: "Down" } }), "Down");
  assert.equal(errorMessage(0, null), "Cannot reach the EasyRAG server.");
  assert.equal(errorMessage(502, "<html>"), "The server answered with HTTP 502.");
});

test("trace summary", () => {
  const trace = {
    total_ms: 4210, retrieval_ms: 312, prompt_tokens: 850, completion_tokens: 42,
    retrieved: [{ chunk_id: "a", score: 0.0312 }], reranked: false, llm_model: "qwen",
  };
  assert.equal(traceSummary(trace), "4.2 s · retrieval 0.31 s · 850 + 42 tokens · top score 0.031 · qwen");
  assert.equal(traceSummary(null), "");
  assert.equal(
    traceSummary({ ...trace, prompt_tokens: null, retrieved: [], llm_model: null }),
    "4.2 s · retrieval 0.31 s",
  );
});

test("readiness problems list only failing checks", () => {
  assert.deepEqual(readinessProblems({ database: "ok", llm_model: "missing", custom: "bad" }), [
    "The chat model is not available yet (it may still be downloading).",
    "custom: bad",
  ]);
});
