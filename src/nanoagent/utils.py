
MAX_LENGTH_TRUNCATE_CONTENT = 2000


def truncate_content(content: str, max_length: int = MAX_LENGTH_TRUNCATE_CONTENT) -> str:
    """Keep short text intact or retain its beginning and ending around a marker.

    Args:
        content: Text to preserve or shorten.
        max_length: Threshold controlling how many source characters are retained;
            half are taken from each end when truncation is needed.

    Returns:
        The original content when its length is within the threshold; otherwise
        a prefix, truncation notice, and suffix. The notice makes the returned
        string longer than ``max_length``.

    Notes:
        ``max_length`` should be a nonnegative integer.
    """
    if len(content) <= max_length:
        return content
    else:
        return (
            content[: max_length // 2]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
            + content[-max_length // 2 :]
        )
