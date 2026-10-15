"""Central configuration for the evidence_rag package."""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
KNOWLEDGE_BASE_DIR = PACKAGE_DIR / "knowledge_base"
VECTOR_DB_DIR = PACKAGE_DIR / "vector_db"
COLLECTION_NAME = "research_abstracts"
ARTICLE_MODEL_NAME = "ncbi/MedCPT-Article-Encoder"
QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"
MAX_CHUNK_SIZE = 2000
TOP_K = 3
