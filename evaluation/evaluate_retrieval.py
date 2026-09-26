"""Measure Recall@5 for the baseline RepoLens retrieval collection."""

import argparse
import json
from pathlib import Path
from typing import Any

from repolens.config import config
from repolens.retrieval.qdrant_store import create_qdrant_client
from repolens.retrieval.search import search_repository


DEFAULT_QUESTIONS_PATH = Path(__file__).with_name("week1_questions.json")


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load and minimally validate retrieval evaluation questions."""

    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError("Evaluation file must contain a non-empty JSON list.")

    for item in questions:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("question"), str)
            or not isinstance(item.get("expected_files"), list)
            or not item["expected_files"]
        ):
            raise ValueError("Each evaluation item needs a question and expected_files.")

    return questions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
    )
    parser.add_argument(
        "--repository",
        default="RepoLens",
        help="Exact repository payload value used during indexing.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=str(config["QDRANT"]["URL"]),
    )
    parser.add_argument(
        "--collection-name",
        default=str(config["QDRANT"]["COLLECTION_NAME"]),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default=str(config["EMBEDDING"]["DEVICE"]),
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)
    client = create_qdrant_client(args.qdrant_url)
    successful = 0

    for number, item in enumerate(questions, start=1):
        results = search_repository(
            item["question"],
            client,
            collection_name=args.collection_name,
            top_k=args.top_k,
            repository=args.repository,
            device=args.device,
        )
        retrieved_files = [result.metadata.relative_path for result in results]
        expected_files = set(item["expected_files"])
        matched = bool(expected_files.intersection(retrieved_files))
        successful += int(matched)

        status = "PASS" if matched else "MISS"
        print(f"{number:02d}. {status} | {item['question']}")
        print(f"    Expected: {', '.join(item['expected_files'])}")
        print(f"    Retrieved: {', '.join(retrieved_files) or '(none)'}")

    recall = successful / len(questions)
    print(f"\nRecall@{args.top_k}: {recall:.2%} ({successful}/{len(questions)})")

    return 0 if recall >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main())