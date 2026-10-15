"""MedCPT document embedding helpers. Models are loaded only when requested."""
from typing import Any

ARTICLE_MODEL_NAME = "ncbi/MedCPT-Article-Encoder"


def load_article_encoder(model_name: str = ARTICLE_MODEL_NAME) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def generate_embedding(text: str, title: str = "", *, tokenizer: Any, model: Any) -> list[float]:
    import torch
    encoded = tokenizer([[title, text]], truncation=True, padding=True,
                        return_tensors="pt", max_length=512)
    with torch.no_grad():
        output = model(**encoded)
    return output.last_hidden_state[:, 0, :].squeeze().cpu().tolist()
