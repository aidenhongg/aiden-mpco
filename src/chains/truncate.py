from ..llm import ENCODING


def truncate_head(text: str, n_tokens: int, *, label: str) -> str:
    """Drop leading tokens; keep the tail.

    Use when the trailing content matters most (e.g. refiner prompts where the
    final instruction follows the prior prompt body).
    """
    tokens = ENCODING.encode(text)
    if len(tokens) <= n_tokens:
        return text
    print(f"[TRUNCATE] {label}: {len(tokens)} -> {n_tokens} tokens (-{len(tokens) - n_tokens})")
    return ENCODING.decode(tokens[-n_tokens:])


def truncate_tail(text: str, n_tokens: int, *, label: str) -> str:
    """Drop trailing tokens; keep the head.

    Use for code snippets where the signature and early body carry the most
    information.
    """
    tokens = ENCODING.encode(text)
    if len(tokens) <= n_tokens:
        return text
    print(f"[TRUNCATE] {label}: {len(tokens)} -> {n_tokens} tokens (-{len(tokens) - n_tokens})")
    return ENCODING.decode(tokens[:n_tokens])
