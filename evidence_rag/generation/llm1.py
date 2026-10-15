"""Groq chat completion using retrieved evidence as the only context."""
import os
import requests


def generate(question: str, evidence: list[str], api_key: str | None = None,
             model: str = "openai/gpt-oss-20b") -> str:
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        return "Groq API key not set; set GROQ_API_KEY to generate an answer."
    context = "\n\n".join(f"Evidence {i + 1}:\n{text}" for i, text in enumerate(evidence))
    response = requests.post("https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "temperature": 0.2, "messages": [
            {"role": "system", "content": "Answer only from the supplied evidence. If it is insufficient, say so."},
            {"role": "user", "content": f"Evidence:\n{context}\n\nQuestion: {question}"}],}, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
