from unstructured.partition.pdf import partition_pdf
from pathlib import Path
import requests
import weaviate

WEAVIATE_HTTP = "http://localhost:8080"
WEAVIATE_GRPC_HOST = "localhost"
WEAVIATE_GRPC_PORT = 50051

# Weaviate (in Docker) -> Ollama (on host)
OLLAMA_ENDPOINT_FROM_WEAVIATE = "http://host.docker.internal:11434"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# Your local Ollama endpoint for *generation* (this Python runs on host)
OLLAMA_GEN_ENDPOINT_FROM_PYTHON = "http://localhost:11434"
OLLAMA_CHAT_MODEL = "llama3.1"  # change to what you have: llama3, mistral, etc.

CLASS_NAME = "FinancesReports"


def ensure_schema():
    schema = {
        "class": CLASS_NAME,
        "vectorizer": "text2vec-ollama",
        "properties": [
            {"name": "category", "dataType": ["text"]},
            {"name": "text", "dataType": ["text"]},
            {"name": "source", "dataType": ["text"]},
        ],
        "moduleConfig": {
            "text2vec-ollama": {
                "model": OLLAMA_EMBED_MODEL,
                "apiEndpoint": OLLAMA_ENDPOINT_FROM_WEAVIATE,
            }
        },
    }

    # Delete if exists (ignore 404)
    requests.delete(f"{WEAVIATE_HTTP}/v1/schema/{CLASS_NAME}")

    r = requests.post(f"{WEAVIATE_HTTP}/v1/schema", json=schema)
    r.raise_for_status()
    print(f"✅ Schema ready: {CLASS_NAME}")


def ingest_pdf(pdf_path: Path):
    elements = partition_pdf(filename=str(pdf_path))
    data_objects = []

    for elem in elements:
        text = (getattr(elem, "text", "") or "").strip()
        if not text:
            continue

        data_objects.append(
            {
                "category": getattr(elem, "category", "Unknown"),
                "text": text,
                "source": pdf_path.name,
            }
        )

    return data_objects

import requests

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_CHAT_MODEL = "llama3:latest"

def ollama_generate(question: str, context: str) -> str:
    prompt = f"""You are a helpful finance document assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

CONTEXT:
{context}

QUESTION:
{question}

Answer:
"""

    url = f"{OLLAMA_BASE}/api/generate"
    resp = requests.post(
        url,
        json={"model": OLLAMA_CHAT_MODEL, "prompt": prompt, "stream": False},
        timeout=180,
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"Ollama error {resp.status_code} for {url}\n{resp.text}"
        )

    return resp.json().get("response", "").strip()



def semantic_search(client, query: str, limit: int = 8):
    """
    Uses Weaviate vector search (nearText).
    """
    col = client.collections.get(CLASS_NAME)

    # v4 search API (works on 4.8.1)
    res = col.query.near_text(
        query=query,
        limit=limit,
        return_properties=["category", "text", "source"],
    )
    return res.objects


def build_context(hits, max_chars: int = 12000) -> str:
    """
    Build a context string from Weaviate hits.
    """
    chunks = []
    total = 0

    for i, obj in enumerate(hits, start=1):
        props = obj.properties or {}
        chunk = (
            f"[{i}] source={props.get('source')} category={props.get('category')}\n"
            f"{props.get('text')}\n"
        )
        if total + len(chunk) > max_chars:
            break
        chunks.append(chunk)
        total += len(chunk)

    return "\n---\n".join(chunks)


def main():
    pdf_path = Path(__file__).resolve().parent / "files" / "bank_financials.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # 1) Ensure schema
    ensure_schema()

    # 2) Ingest
    data_objects = ingest_pdf(pdf_path)
    print(f"Extracted {len(data_objects)} elements")

    client = weaviate.connect_to_local(
        host=WEAVIATE_GRPC_HOST,
        port=8080,
        grpc_port=WEAVIATE_GRPC_PORT,
    )

    try:
        col = client.collections.get(CLASS_NAME)

        with col.batch.fixed_size(batch_size=200) as batch:
            for obj in data_objects:
                batch.add_object(properties=obj)

        failed = col.batch.failed_objects
        if failed:
            print(f"⚠️ Failed imports: {len(failed)} (showing first)")
            print(failed[0])
        else:
            print(f"✅ Imported {len(data_objects)} objects into {CLASS_NAME}")

        # 3) Semantic search + RAG
        question = "What is the company's net profit and total revenue?"
        hits = semantic_search(client, question, limit=8)
        print(f"Found {len(hits)} hits")

        context = build_context(hits)
        answer = ollama_generate(question, context)

        print("\n=== QUESTION ===")
        print(question)
        print("\n=== ANSWER ===")
        print(answer)

    finally:
        client.close()


if __name__ == "__main__":
    main()

