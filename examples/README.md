# Examples

## A sample corpus: `docs/`

Four short pages from the wiki of **Orbit**, a fictional notification service. Because
Orbit is invented, every correct answer has to come from these files. Use the corpus to
try EasyRAG before pointing it at your own documents:

```bash
echo "EASYRAG_DOCS_HOST_DIR=./examples/docs" >> .env
docker compose up -d
```

Then open http://localhost:8000 and try these questions:

| Question | What to expect |
|---|---|
| How do I roll back a production deploy? | `make rollback ENV=production`, cited to *deployment.md › Rollback* |
| On which days are production deploys allowed? | Tuesdays and Thursdays |
| Why was my SMS not delivered? | Quiet hours, cited to the FAQ |
| How many times does Orbit retry a failed delivery? | Five, with backoff from 30 seconds |
| What should I check before promoting the replica? | Replication lag above 60 seconds: escalate instead |
| What does `ORBIT_QUIET_HOURS` default to? | `22:00-07:00` (from the configuration table) |
| Which SMS providers are supported? | `twilio` or `vonage` |
| How much does Orbit cost per message? | *I don't know based on the provided documents.* (not in the corpus) |

The corpus shows what EasyRAG handles: headings at several levels, a subfolder
(`runbooks/`), fenced code blocks, a table and bullet lists. Every answer links to the
exact section and line range.

To go back to your own documents, remove the line from `.env` and run
`docker compose up -d` again. The index follows automatically: removed files leave the
index, new ones are added.

## API clients: `clients/`

Both read `EASYRAG_URL` (default `http://localhost:8000`) and, if your server requires
one, `EASYRAG_API_KEY`.

```bash
python3 examples/clients/ask.py "How do I roll back a production deploy?"
```

```bash
examples/clients/ask.sh "Who owns Orbit?"
```

`ask.py` uses only the Python standard library. Its exit code is 0 when the question was
answered, 1 when the documents don't contain the answer, and 2 on errors, so it can be
used in scripts. `ask.sh` needs `curl` and `jq`.

## Evaluating your own documents

`eval/datasets/nimbus/` is a complete evaluation dataset: a corpus plus `questions.yaml`
with expected sources and key facts. Copy its layout to measure retrieval and answer
quality on your own documents (see [Evaluating quality](../README.md#evaluating-quality)).
