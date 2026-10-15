"""Interactive query entry point: run with ``python -m evidence_rag.query``."""
import chromadb
from dotenv import load_dotenv
from . import config
from .generation.llm1 import generate
from .retrieval.medcpt import load_query_encoder, search


def main() -> None:
    load_dotenv(config.PROJECT_DIR / ".env")
    tokenizer, model = load_query_encoder(config.QUERY_MODEL_NAME)
    client = chromadb.PersistentClient(path=str(config.VECTOR_DB_DIR))
    collection = client.get_collection(name=config.COLLECTION_NAME)
    print(f"Connected to {config.COLLECTION_NAME} ({collection.count()} documents)")
    while True:
        question = input("Your question (or 'quit'): ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        results = search(collection, question, tokenizer, model, config.TOP_K)
        documents = results.get("documents", [[]])[0]
        for index, document in enumerate(documents, 1):
            print(f"\nEvidence {index}: {document}")
        print("\nAnswer:\n" + generate(question, documents))


if __name__ == "__main__":
    main()
