"""Qdrant collection management and embedded-chunk persistence."""

from collections.abc import Sequence
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from repolens.config import config, logger
from repolens.models import ChunkMetadata, EmbeddedChunk, SearchResult


def create_qdrant_client(
    url: str = str(config["QDRANT"]["URL"]),
) -> QdrantClient:
    """Create a Qdrant client."""

    if not url.strip():
        raise ValueError("Qdrant URL is not set in the configuration.")

    return QdrantClient(url=url)


def _resolve_distance(distance: str) -> models.Distance:
    """Convert a configured distance name to Qdrant's enum."""

    normalized = distance.strip().lower()
    distances = {
        "cosine": models.Distance.COSINE,
        "dot": models.Distance.DOT,
        "euclid": models.Distance.EUCLID,
        "manhattan": models.Distance.MANHATTAN,
    }

    try:
        return distances[normalized]
    except KeyError as error:
        supported = ", ".join(sorted(distances))
        raise ValueError(
            f"Unsupported distance '{distance}'. Supported distances are: {supported}."
        ) from error


def point_id_for_chunk(chunk_id: str) -> str:
    """Generate a unique point ID for a chunk using UUID5."""

    if not chunk_id:
        raise ValueError("Chunk ID cannot be empty.")

    return str(uuid5(NAMESPACE_URL, f"repolens:{chunk_id}"))


def _collection_vector_params(collection_info: Any) -> Any:
     """Return the unnamed-vector parameters from collection information."""

     vectors_config = collection_info.config.params.vectors

     if isinstance(vectors_config, dict):
         raise ValueError("RepoLens requires a Qdrant collection with one unnamed vector.")

     return vectors_config


def ensure_collection(
    client: QdrantClient,
    vector_size: int,
    *,
    collection_name: str = str(config["QDRANT"]["COLLECTION_NAME"]),
    distance: str = str(config["QDRANT"]["DISTANCE"]),
) -> bool:
    """Create the collection if absent and validate it if already present.

    Returns ``True`` when a new collection is created and ``False`` when an
    existing compatible collection is reused.
    """

    if not collection_name.strip():
        raise ValueError("Qdrant collection name cannot be empty.")
    if vector_size <= 0:
        raise ValueError("Vector size must be greater than zero.")

    expected_distance = _resolve_distance(distance)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size, 
                distance=expected_distance,
            ),
        )
        logger.info(
            "Created Qdrant collection '%s' with %d-dimensional vectors.",
            collection_name,
            vector_size,
        )
        return True

    collection_info = client.get_collection(collection_name)
    vector_params = _collection_vector_params(collection_info)

    actual_size = int(vector_params.size)
    actual_distance = vector_params.distance

    if actual_size != vector_size or actual_distance != expected_distance:
        raise ValueError(
            f"Qdrant collection '{collection_name}' is incompatible: "
            f"expected size={vector_size}, distance={expected_distance.value}; "
            f"found size={actual_size}, distance={actual_distance.value}."
        )

    return False


def _chunk_payload(chunk: EmbeddedChunk) -> dict[str, object]:
    """Convert an EmbeddedChunk to a Qdrant payload dictionary."""

    metadata = chunk.metadata

    return {
        "content": chunk.content,
        "repository": metadata.repository,
        "relative_path": metadata.relative_path,
        "file_type": metadata.file_type,
        "start_line": metadata.start_line,
        "end_line": metadata.end_line,
        "token_count": metadata.token_count,
        "content_hash": metadata.content_hash,
        "chunk_id": metadata.chunk_id,
        "heading": metadata.heading,
    }


def upsert_embedded_chunks(
    client: QdrantClient,
    embedded_chunks: Sequence[EmbeddedChunk],
    *,
    collection_name: str = str(config["QDRANT"]["COLLECTION_NAME"]),
    distance: str = str(config["QDRANT"]["DISTANCE"]),
    batch_size: int = int(config["QDRANT"]["UPSERT_BATCH_SIZE"]),
) -> int:
    """Create/validate a collection and idempotently upsert chunks."""

    if batch_size <= 0:
        raise ValueError("Batch size must be greater than zero.")
    if not embedded_chunks:
        logger.info("No embedded chunks were provided for indexing.")
        return 0

    dimensions = {len(chunk.embedding) for chunk in embedded_chunks}

    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError("Embedded chunks must contain non-empty vectors with one dimension.")

    vector_size = next(iter(dimensions))
    ensure_collection(
        client, 
        vector_size, 
        collection_name=collection_name, 
        distance=distance,
    )

    points = [
        models.PointStruct(
            id=point_id_for_chunk(chunk.metadata.chunk_id),
            vector=chunk.embedding,
            payload=_chunk_payload(chunk),
        )
        for chunk in embedded_chunks
    ]

    for start in range(0, len(points), batch_size):
        client.upsert(
            collection_name=collection_name,
            points=points[start:start + batch_size],
            wait=True,
        )

    logger.info(
        "Indexed %d chunks into Qdrant collection '%s'.",
        len(points),
        collection_name,
    )

    return len(points)


def build_search_filter(
    *,
    repository: str | None = None,
    file_type: str | None = None,
) -> models.Filter | None:
    """Build an optional exact-match Qdrant payload filter."""

    conditions: list[models.FieldCondition] = []

    if repository is not None:
        repository = repository.strip()
        if not repository:
            raise ValueError("Repository filter cannot be empty.")
        conditions.append(
            models.FieldCondition(
                key="repository",
                match=models.MatchValue(value=repository),
            )
        )

    if file_type is not None:
        file_type = file_type.strip()
        if not file_type:
            raise ValueError("File_type filter cannot be empty.")
        conditions.append(
            models.FieldCondition(
                key="file_type",
                match=models.MatchValue(value=file_type),
            )
        )

    if not conditions:
        return None

    return models.Filter(must=conditions)


def _metadata_from_payload(payload: dict[str, Any]) -> ChunkMetadata:
    """Rebuild validated chunk metadata from a Qdrant payload."""

    required_fields = (
        "repository",
        "relative_path",
        "file_type",
        "start_line",
        "end_line",
        "token_count",
        "content_hash",
        "chunk_id",
    )

    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        raise ValueError(
            "Qdrant result payload is missing fields: "
            + ", ".join(missing_fields)
        )

    heading = payload.get("heading")

    return ChunkMetadata(
        repository=str(payload["repository"]),
        relative_path=str(payload["relative_path"]),
        file_type=str(payload["file_type"]),
        start_line=int(payload["start_line"]),
        end_line=int(payload["end_line"]),
        token_count=int(payload["token_count"]),
        content_hash=str(payload["content_hash"]),
        chunk_id=str(payload["chunk_id"]),
        heading=str(heading) if heading is not None else None,
    )


def search_points(
    client: QdrantClient,
    query_vector: Sequence[float],
    *,
    collection_name: str = str(config["QDRANT"]["COLLECTION_NAME"]),
    distance: str = str(config["QDRANT"]["DISTANCE"]),
    top_k: int = int(config["RETRIEVAL"]["TOP_K"]),
    repository: str | None = None,
    file_type: str | None = None,
) -> list[SearchResult]:
    """Run a filtered nearest-neighbour query and return ranked results."""

    if not query_vector:
        raise ValueError("Query vector cannot be empty.")
    if top_k <= 0:
        raise ValueError("Top K must be greater than zero.")

    query_filter = build_search_filter(
        repository=repository, 
        file_type=file_type,
    )

    response = client.query_points(
        collection_name=collection_name,
        query=[float(value) for value in query_vector],
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    results: list[SearchResult] = []

    for rank, point in enumerate(response.points, start=1):
        payload = dict(point.payload or {})

        if "content" not in payload:
            raise ValueError("Qdrant result payload is missing field: content")

        results.append(
            SearchResult(
                rank=rank,
                score=float(point.score),
                content=str(payload["content"]),
                metadata=_metadata_from_payload(payload),
            )
        )

    return results


