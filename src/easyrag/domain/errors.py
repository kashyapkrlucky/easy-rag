"""Domain errors shared across layers."""


class IndexIncompatibleError(Exception):
    """The existing index was built with a different embedding model or dimension."""

    def __init__(
        self,
        *,
        stored_model: str,
        stored_dimension: int,
        configured_model: str,
        configured_dimension: int,
    ) -> None:
        super().__init__(
            f"The index was built with {stored_model} ({stored_dimension} dimensions), but the "
            f"configured embedding model is {configured_model} ({configured_dimension} "
            "dimensions). Vectors from different models cannot be mixed: run "
            "`easyrag index --rebuild` to re-index everything with the configured model."
        )
        self.stored_model = stored_model
        self.stored_dimension = stored_dimension
