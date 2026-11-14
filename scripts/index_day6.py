"""Rebuild the Day 6-compatible paragraph vectors in PostgreSQL/pgvector."""
from src.retrieval import MODEL_NAME, RetrievalSystem


def main() -> None:
    system = RetrievalSystem()
    count = system.index()
    print(f"Indexed {count} chunks using {MODEL_NAME} into strategy C")


if __name__ == "__main__":
    main()
