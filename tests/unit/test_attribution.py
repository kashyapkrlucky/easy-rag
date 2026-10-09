from easyrag.domain.chunks import RetrievedChunk
from easyrag.rag.attribution import attribute


def chunk(chunk_id: str, content: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id, f"{chunk_id}.md", chunk_id, (), "", content, 1, 1, 0.5)


DEPLOY = chunk(
    "deploy",
    "1. Copy `.env.example` to `.env` and set `POSTGRES_PASSWORD`.\n"
    "2. Put your Markdown files in `./docs`.\n"
    "3. Run `docker compose up -d` and wait for `/health/ready` to report ready.",
)
SUPPORT = chunk(
    "support", "Support requests go to the internal helpdesk. Response time is one business day."
)
MODEL = chunk("model", "The default embedding model is nomic-embed-text:v1.5 with 768 dimensions.")


def test_answer_copied_from_a_source_is_supported() -> None:
    result = attribute("The support response time is one business day.", [DEPLOY, SUPPORT])

    assert result.supported
    assert [s.chunk_id for s in result.sources] == ["support"]
    assert result.coverage == 1.0


def test_multi_sentence_answer_is_attributed_to_each_supporting_source_in_order() -> None:
    answer = (
        "Run docker compose up and wait for the health endpoint to report ready.\n"
        "Support response time is one business day."
    )

    result = attribute(answer, [SUPPORT, DEPLOY])

    assert [s.chunk_id for s in result.sources] == ["deploy", "support"]  # order of use


def test_answer_with_content_absent_from_the_sources_is_unsupported() -> None:
    result = attribute("Paris is the capital of France.", [DEPLOY, SUPPORT, MODEL])

    assert not result.supported
    assert result.sources == ()
    assert result.coverage == 0.0


def test_mostly_supported_answer_with_one_invented_claim_passes_the_coverage_threshold() -> None:
    answer = (
        "Copy the env example file and set the Postgres password. Put Markdown files in docs. "
        "Run docker compose up and wait until the health endpoint reports ready. "
        "Kubernetes works too."
    )

    result = attribute(answer, [DEPLOY])

    assert result.supported
    assert 0.7 <= result.coverage < 1.0


def test_answer_dominated_by_invented_claims_is_unsupported() -> None:
    answer = (
        "Run docker compose up. "
        "Then configure the Kubernetes autoscaler, the Helm chart, the ingress controller "
        "and the certificate manager for production clusters."
    )

    assert not attribute(answer, [DEPLOY]).supported


def test_numbers_and_identifiers_count_as_content() -> None:
    assert attribute(
        "The default model is nomic-embed-text:v1.5 with 768 dimensions.", [MODEL]
    ).supported
    assert not attribute("The default model is all-minilm with 384 dimensions.", [MODEL]).supported


def test_citation_markers_and_stopwords_are_ignored() -> None:
    assert attribute("Response time is one business day [3].", [SUPPORT]).supported
    assert not attribute("It is what it is.", [SUPPORT]).supported  # no content words


def test_empty_inputs_are_unsupported() -> None:
    assert not attribute("", [SUPPORT]).supported
    assert not attribute("One business day.", []).supported
