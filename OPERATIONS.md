# Operations and troubleshooting

Day-to-day commands, what to monitor, and fixes by symptom. All commands are run from the
EasyRAG directory. If you set `EASYRAG_API_KEY`, add `-H "X-API-Key: $KEY"` to the `curl`
calls under `/api/` and `/metrics`.

## Is it working?

```bash
docker compose ps                                   # every service "running (healthy)"
curl -s localhost:8000/health/ready                 # {"status":"ready","checks":{... all "ok"}}
curl -s localhost:8000/api/v1/index/status          # state, last report, document/chunk counts
```

| Readiness check | Value | Meaning |
|---|---|---|
| `database` | `unavailable` | PostgreSQL is down or unreachable |
| `schema` | `outdated` | Migrations pending. They run when the app container starts, so restart it |
| `embedding_model`, `llm_model` | `pulling` | Downloading (first start, or the model was changed) |
| | `missing` | Not in Ollama and not being pulled. See *Models don't download* below |
| | `misconfigured` | Embedding: its output size isn't `EASYRAG_EMBEDDING_DIMENSION`. Chat: its context is smaller than `EASYRAG_LLM_CONTEXT_WINDOW` |
| | `unavailable` | Ollama is unreachable |
| `index` | `misconfigured` | The index was built with another embedding model. Rebuild it |
| `reranker` | `missing` | Only with reranking on: the model is still loading. Answers work meanwhile |

## Logs

```bash
docker compose logs -f app                          # JSON lines
docker compose logs app | grep '"level": "WARNING"'
docker compose logs app | grep '"request_id": "<id>"'   # one request; the id is in X-Request-ID
```

- Every response carries `X-Request-ID`, and the web UI shows it on errors. All log lines
  for that request share it.
- Index runs share an `index_run_id` instead.
- For readable logs while debugging, set `EASYRAG_LOG_FORMAT=text`.
- `EASYRAG_LOG_LEVEL=DEBUG` also logs questions and unverified answer sentences. **Don't
  leave DEBUG on in production.**

## Metrics

`GET /metrics` serves Prometheus metrics. Suggested alerts:

| Alert | Expression (PromQL) |
|---|---|
| Answers failing | `rate(easyrag_api_errors_total{code=~"llm_.*\|embedding_.*\|database_.*"}[5m]) > 0` |
| Questions turned away | `rate(easyrag_api_errors_total{code="busy"}[15m]) > 0` |
| Indexing failing | `increase(easyrag_index_runs_total{result="failed"}[1h]) > 0` |
| Index stale | `time() - easyrag_index_last_success_timestamp_seconds > 86400` (with the 5-minute resync, a day without success means trouble) |
| Slow answers | `histogram_quantile(0.95, rate(easyrag_rag_stage_duration_seconds_bucket{stage="total"}[30m])) > 60` |
| Injection attempts | `increase(easyrag_rag_redacted_sources_total[1d]) > 0`: some document contains instruction-like text. Find it in the WARNING logs |
| Many unverified answers | `rate(easyrag_rag_unverified_answers_total[1d]) / rate(easyrag_rag_answers_total[1d]) > 0.2` |

## Common tasks

| Task | Command |
|---|---|
| Re-index now (incremental) | `curl -X POST localhost:8000/api/v1/index` |
| Rebuild the index from scratch | `docker compose run --rm app easyrag index --rebuild` |
| Change the documents folder | Set `EASYRAG_DOCS_HOST_DIR` in `.env`, then `docker compose up -d` |
| Change the chat model | Set `EASYRAG_LLM_MODEL` (and maybe `EASYRAG_OLLAMA_MEMORY_LIMIT`), then `docker compose up -d`. It's pulled automatically |
| Change the embedding model | Set the model, dimension and prefixes (see `.env.example`), then `docker compose up -d`, then rebuild the index |
| Check answer quality after a change | `easyrag eval` (see [BENCHMARKS.md](BENCHMARKS.md)) |
| See what the model was given for one answer | Ask with `"debug": true`. The `trace` lists the retrieved and cited chunk ids, scores, tokens and timings |
| Upgrade | `git pull && docker compose up -d --build` (see *Upgrading* in the README) |
| Reset everything except models | `docker compose down && docker volume rm easyrag_pgdata easyrag_secrets && docker compose up -d` |

## Troubleshooting

### `docker compose up` fails
- **`depends on undefined service` or `profiles` errors:** you need Compose v2.24 or later
  (`docker compose version`).
- **`port is already allocated`:** something else uses port 8000. Set `EASYRAG_PORT`.
- **`init` exits with an error:** check its output with `docker compose logs init`.

### The app can't connect to the database (`password authentication failed`)
The database was created with a different password than the one now in the `secrets`
volume. This happens if `POSTGRES_PASSWORD` was removed from `.env` after the database had
been created with it, or if only the `secrets` volume was deleted.
- Put the original `POSTGRES_PASSWORD` back in `.env`, and run `docker compose up -d`.
- Or reset the index, which is safe because it's rebuilt from your documents: delete the
  `easyrag_pgdata` and `easyrag_secrets` volumes (see *Common tasks*).

### Models don't download (`missing` or `pulling` for a long time)
- **Watch progress:** `docker compose logs -f app | grep -i pull`.
- **Offline, or a proxy:** Ollama needs internet access to `registry.ollama.ai`. Otherwise
  copy the models into Ollama yourself and set `EASYRAG_OLLAMA_PULL_MODELS=false`.
- **Model name typo:** the log says `pulling '<model>' failed: file does not exist`. Fix
  the tag in `.env`.
- **Host Ollama on Linux:** Ollama must listen where containers can reach it
  (`OLLAMA_HOST=0.0.0.0`). Check with
  `docker compose exec app python -c "import urllib.request as u; print(u.urlopen('http://host.docker.internal:11434').read())"`.

### Documents aren't indexed, or changes aren't picked up
- **Check the status:** `curl -s localhost:8000/api/v1/index/status`.
  - `failed` / `skipped` list files and reasons: unreadable, not UTF-8, or too large.
  - `deletion_guarded: true` means the docs folder looked empty, so nothing was deleted.
    Check `EASYRAG_DOCS_HOST_DIR` and that the folder is shared with Docker. On Docker
    Desktop: *Settings → Resources → File sharing*.
- **Only `*.md` and `*.markdown` files are indexed.** Hidden files and folders are skipped.
- **Changes not noticed** on network filesystems or some bind mounts: set
  `EASYRAG_WATCH_FORCE_POLLING=true`. A full incremental check also runs every 5 minutes
  anyway.

### Answers are "I don't know" too often
1. **Check retrieval:** `curl -s localhost:8000/api/v1/retrieve -H 'Content-Type: application/json' -d '{"query": "<your question>"}'`.
   If the right section isn't among the results, retrieval is the problem:
   - use the words your documents use;
   - split very long sections, or use clearer headings (headings are part of every chunk);
   - try `"mode": "vector"` or `"rerank": true` to compare.
2. **If the right section is retrieved,** the model didn't answer from it, or its answer
   couldn't be verified (`"debug": true` shows `unverified_sentences`):
   - a larger chat model helps most;
   - `EASYRAG_RAG_UNGROUNDED_POLICY=flag` returns unverified answers marked
     `"grounded": false`, instead of refusing them.
3. **Is `EASYRAG_RETRIEVAL_MIN_SCORE` set?** It's off by default; a high value refuses
   answerable questions.

### Answers are wrong, or include something they shouldn't
- **Open the cited source** (click `[n]` in the web UI). If the source says it, the
  document is the problem.
- **A source shows `[removed: instruction-like text]`** in the model's view, and a WARNING
  names the document: that document contains text aimed at the model. Review it.
- **Strange answers that ignore your question:** possible prompt injection in a document
  that isn't caught by the patterns. Search the cited sources for instructions, and remove
  or fix the document.
- **Report reproducible cases** with the question, the debug trace and the cited sources.

### Answers are slow
- **The first answer after a pause loads the model** (about 10 s on CPU).
  `OLLAMA_KEEP_ALIVE` (default 1 hour) controls how long it stays loaded.
- **Prompt size drives CPU time:** lower `EASYRAG_RAG_MAX_CONTEXT_TOKENS` or
  `EASYRAG_RETRIEVAL_TOP_K`, and compare quality with `easyrag eval`.
- **On macOS,** use a native Ollama: Docker can't use the GPU (see the README).
- **`503 busy`:** more than `EASYRAG_CHAT_MAX_CONCURRENT` questions are being answered.
  Ollama answers one at a time by default (`OLLAMA_NUM_PARALLEL=1`).

### A container keeps restarting
- Check `docker compose ps` and `docker inspect <container> --format '{{.State.ExitCode}} {{.State.OOMKilled}}'`.
- **Exit code 137 with `OOMKilled: true`:** raise its memory limit in `.env`
  (`EASYRAG_OLLAMA_MEMORY_LIMIT` for a bigger model), or use a smaller model. Docker
  Desktop's VM memory is under *Settings → Resources*.
- **Ollama logs `model requires more system memory`:** same fix.

### `401 unauthorized` / `413` / `index_incompatible`
- **`401`:** send the key as `X-API-Key` or `Authorization: Bearer`. The web UI asks for it.
  A key in `.env` takes effect after `docker compose up -d`.
- **`413`:** the request body is over `EASYRAG_MAX_REQUEST_BYTES` (64 KiB).
- **`index_incompatible`:** the embedding settings changed since the index was built.
  Rebuild the index.
