import json
import math
from pathlib import Path

from repolens.config import logger


_CACHE_VERSION = 1


def load_embedding_cache(
    cache_path: Path,
    *,
    model_name: str,
    normalize: bool,
) -> dict[str, list[float]]:
    """Load compatible embedding cache from disk."""

    if not cache_path.exists():
        logger.debug("Embedding cache file '%s' does not exist.", cache_path)
        return {}

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Embedding cache is not valid JSON: {cache_path}"
        ) from error
    except OSError as error:
        raise ValueError(
            f"Unable to read embedding cache: {cache_path}"
        ) from error

    if not isinstance(payload, dict):
        raise ValueError(
            f"Embedding cache must contain a JSON object: {cache_path}"
        )

    is_compatible = (
        payload.get("version") == _CACHE_VERSION
        and payload.get("model_name") == model_name
        and payload.get("normalize") == normalize
    )

    if not is_compatible:
        logger.info(
            "Ignoring incompatible embedding cache: %s",
            cache_path,
        )
        return {}

    raw_embeddings = payload.get("embeddings")

    if not isinstance(raw_embeddings, dict):
        raise ValueError(
            "Embedding cache is missing its embeddings object"
        )

    embeddings: dict[str, list[float]] = {}

    for chunk_id, vector in raw_embeddings.items():
        if(
            not isinstance(chunk_id, str)
            or not isinstance(vector, list)
            or not vector
        ):
            raise ValueError(
                "Embedding cache contains an invalid entry"
            )

        try:
            converted_vector = [float(value) for value in vector]
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid cached vector for chunk {chunk_id}"
            ) from error

        if not all(
            math.isfinite(value)
            for value in converted_vector
        ):
            raise ValueError(
                f"Cached vector contains non-finite values: "
                f"{chunk_id}"
            )

        embeddings[chunk_id] = converted_vector

    dimensions = {len(vector) for vector in embeddings.values()}

    if len(dimensions) > 1:
        raise ValueError(
            "Cached embeddings have inconsistent dimensions"
        )

    return embeddings


def save_embedding_cache(
    cache_path: Path,
    embeddings: dict[str, list[float]],
    *,
    model_name: str,
    normalize: bool,
) -> None:
    """Save embedding cache to disk."""

    payload = {
        "version": _CACHE_VERSION,
        "model_name": model_name,
        "normalize": normalize,
        "embeddings": embeddings,
    }

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = cache_path.with_name(
        f"{cache_path.name}.tmp"
    )

    try:
        temporary_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(cache_path)
    except OSError as error:
        raise RuntimeError(
            f"Unable to save embedding cache: {cache_path}"
        ) from error
    finally:
        temporary_path.unlink(missing_ok=True)