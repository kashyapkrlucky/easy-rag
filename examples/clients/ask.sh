#!/usr/bin/env sh
# Ask EasyRAG a question with curl.     ./ask.sh "Who owns Orbit?"
# Environment: EASYRAG_URL (default http://localhost:8000), EASYRAG_API_KEY (if required).
# Needs jq (to build the request safely and print the answer). Exit code 1 on API errors.
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: $0 \"question\"" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required (https://jqlang.org)" >&2
  exit 2
fi

url="${EASYRAG_URL:-http://localhost:8000}/api/v1/chat"
# jq escapes quotes, backslashes and newlines in the question.
body=$(jq -n --arg question "$1" '{question: $question}')

# POSIX sh has no arrays: the optional header goes into the positional parameters.
if [ -n "${EASYRAG_API_KEY:-}" ]; then
  set -- -H "X-API-Key: $EASYRAG_API_KEY"
else
  set --
fi

curl -sS --max-time 300 "$url" -H 'Content-Type: application/json' "$@" -d "$body" | jq -r '
  if .error then ("error: \(.error.message)\n" | halt_error(1))
  else .answer, (.sources | to_entries[] | "  [\(.key + 1)] \(.value.document) > \(.value.section)")
  end'
