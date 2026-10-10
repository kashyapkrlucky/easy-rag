#!/usr/bin/env python3
"""Ask EasyRAG a question from the command line (Python standard library only).

    python3 ask.py "How do I roll back a production deploy?"

Environment: EASYRAG_URL (default http://localhost:8000) and EASYRAG_API_KEY (if the
server requires one). Exit codes: 0 answered, 1 not in the documents, 2 error.
"""

import json
import os
import sys
import urllib.error
import urllib.request


def ask(question: str) -> int:
    url = os.environ.get("EASYRAG_URL", "http://localhost:8000").rstrip("/") + "/api/v1/chat"
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("EASYRAG_API_KEY"):
        headers["X-API-Key"] = key
    request = urllib.request.Request(
        url, data=json.dumps({"question": question}).encode(), headers=headers, method="POST"
    )
    try:
        # Generation on CPU can take a while; match the server's own timeout.
        with urllib.request.urlopen(request, timeout=300) as response:
            answer = json.load(response)
    except urllib.error.HTTPError as error:
        try:
            message = json.load(error)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = error.reason
        print(f"error: HTTP {error.code}: {message}", file=sys.stderr)
        return 2
    except urllib.error.URLError as error:
        print(f"error: cannot reach {url}: {error.reason}", file=sys.stderr)
        return 2

    print(answer["answer"])
    if not answer["grounded"]:
        print("\n(warning: this answer is not verified against your documents)")
    for number, source in enumerate(answer["sources"], start=1):
        start, end = source["lines"]
        section = " > ".join(source["heading_path"])
        print(f"  [{number}] {source['document']} > {section} (lines {start}-{end})")
    return 0 if answer["status"] == "answered" else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    sys.exit(ask(sys.argv[1]))
