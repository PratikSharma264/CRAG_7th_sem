"""MedCPT query encoding and Chroma dense retrieval."""
from typing import Any

QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"


def load_query_encoder(model_name: str = QUERY_MODEL_NAME) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def encode_query(question: str, tokenizer: Any, model: Any) -> list[float]:
    import torch
    encoded = tokenizer([question], truncation=True, padding=True,
                        return_tensors="pt", max_length=512)
    with torch.no_grad():
        output = model(**encoded)
    return output.last_hidden_state[:, 0, :].squeeze().cpu().tolist()


def search(collection: Any, question: str, tokenizer: Any, model: Any, top_k: int = 3) -> dict:
    return collection.query(query_embeddings=[encode_query(question, tokenizer, model)],
                            n_results=top_k)
